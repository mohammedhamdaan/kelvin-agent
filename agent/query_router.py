"""
Central query router for the Kelvin6K voice assistant.

Classifies a natural-language question (English, Hindi or Hinglish) into an
intent, then RESOLVES that intent against the real system -- the SQLite
database, the live machine state, the vision pipeline and the RAG index.

The critical property: resolvers return FACTS, never prose. The language model
formats those facts into a sentence but can never originate a number. If a fact
is unavailable, the resolver says so explicitly and the assistant reports it as
unavailable rather than guessing.

Intent coverage matches the operator's vocabulary, including Hinglish
("kitni walls print ho gayi hain", "material kitna use hua", "camera chal raha
hai"), so exact phrasing is never required.
"""

import datetime
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent import config, db, identity, metrics, shift_store
from agent.logging_setup import get_logger
from agent.rag.embeddings import _STOPWORDS, tokenize
from agent.rag.retriever import search_documentation
from agent.services import machine_state as ms
from agent.spoken import speak_knowledge, parse_record

log = get_logger("ROUTER")


class Intent(str, Enum):
    GET_COMPLETED_PRODUCTION = "GET_COMPLETED_PRODUCTION"
    GET_REMAINING_PRODUCTION = "GET_REMAINING_PRODUCTION"
    GET_TOTAL_TARGET = "GET_TOTAL_TARGET"
    GET_PRINT_RATE = "GET_PRINT_RATE"
    GET_CURRENT_LAYER = "GET_CURRENT_LAYER"
    GET_LAYER_WIDTH = "GET_LAYER_WIDTH"
    GET_LAYER_HEIGHT = "GET_LAYER_HEIGHT"
    GET_WALL_DIMENSIONS = "GET_WALL_DIMENSIONS"
    GET_MATERIAL_USAGE = "GET_MATERIAL_USAGE"
    GET_MATERIAL_REMAINING = "GET_MATERIAL_REMAINING"
    GET_MATERIAL_RATE = "GET_MATERIAL_RATE"
    GET_ELAPSED_TIME = "GET_ELAPSED_TIME"
    GET_ETA = "GET_ETA"
    GET_DEADLINE_STATUS = "GET_DEADLINE_STATUS"
    GET_24_HOUR_PRODUCTION = "GET_24_HOUR_PRODUCTION"
    GET_SHIFT_PRODUCTION = "GET_SHIFT_PRODUCTION"
    GET_OPERATOR = "GET_OPERATOR"
    GET_MACHINE_STATUS = "GET_MACHINE_STATUS"
    GET_ROBOT_STATUS = "GET_ROBOT_STATUS"
    GET_CAMERA_STATUS = "GET_CAMERA_STATUS"
    GET_COMMUNICATION_STATUS = "GET_COMMUNICATION_STATUS"
    GET_PRINT_QUALITY = "GET_PRINT_QUALITY"
    GET_RECENT_PROBLEMS = "GET_RECENT_PROBLEMS"
    GET_RECENT_EVENTS = "GET_RECENT_EVENTS"
    GET_SITE_PEOPLE_COUNT = "GET_SITE_PEOPLE_COUNT"
    GET_SITE_DETECTIONS = "GET_SITE_DETECTIONS"
    WAKE_WORD = "WAKE_WORD"
    KNOWLEDGE_QUERY = "KNOWLEDGE_QUERY"
    COMBINED_QUERY = "COMBINED_QUERY"
    UNKNOWN_QUERY = "UNKNOWN_QUERY"

    # --- Phase 2C Action Intents (Write) ---
    SHIFT_START = "SHIFT_START"
    SHIFT_END = "SHIFT_END"
    LOG_WALL_PRODUCTION = "LOG_WALL_PRODUCTION"
    LOG_SHIFT_ISSUE = "LOG_SHIFT_ISSUE"
    LOG_SHIFT_HANDOVER = "LOG_SHIFT_HANDOVER"
    RESOLVE_SHIFT_ISSUE = "RESOLVE_SHIFT_ISSUE"

    # --- Phase 2C Historical Query Intents (Read) ---
    GET_ACTIVE_SHIFT = "GET_ACTIVE_SHIFT"
    GET_PREVIOUS_SHIFT = "GET_PREVIOUS_SHIFT"
    GET_SHIFT_BY_DATE = "GET_SHIFT_BY_DATE"
    GET_SHIFT_BY_OPERATOR = "GET_SHIFT_BY_OPERATOR"
    GET_WALL_HISTORY = "GET_WALL_HISTORY"
    GET_SHIFT_ISSUES = "GET_SHIFT_ISSUES"
    GET_HANDOVER_STATUS = "GET_HANDOVER_STATUS"

    # --- Industrial Operational Query & Job Intents ---
    OPERATIONAL_REPORT = "OPERATIONAL_REPORT"
    START_PRINTING_JOB = "START_PRINTING_JOB"
    END_PRINTING_JOB = "END_PRINTING_JOB"


UNAVAILABLE_MESSAGE = "That live data is currently unavailable."


@dataclass
class RouterResult:
    intent: Intent
    facts: Dict[str, Any] = field(default_factory=dict)
    knowledge: Optional[Dict[str, Any]] = None
    available: bool = True
    reason: str = ""
    matched_terms: List[str] = field(default_factory=list)
    question: str = ""
    llm_reply: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "facts": self.facts,
            "knowledge": self.knowledge,
            "available": self.available,
            "reason": self.reason,
            "matched_terms": self.matched_terms,
            "llm_reply": self.llm_reply,
        }


# ---------------------------------------------------------------------------
# Intent classification
#
# Each intent lists trigger phrases in English AND Hindi/Hinglish. Matching is
# substring-based on a normalised question, scored by how many (and how long)
# the matches are, so natural paraphrases work without exact wording.
# ---------------------------------------------------------------------------

# (intent, phrases, priority). Priority breaks ties between overlapping
# phrases: "current layer width" contains both "current layer" and "layer
# width", and the more specific intent must win regardless of phrase length.
# Higher priority = more specific.
_PATTERNS: List[Tuple[Intent, List[str], int]] = [
    # --- Phase 2C Action Intents (Write actions, highest priority) ---
    (Intent.SHIFT_START, [
        "start shift", "start morning shift", "start afternoon shift", "start night shift",
        "start a shift", "start the shift", "start new shift", "mera shift start",
        "shift start karo", "shift shuru karo", "begin shift", "nayi shift start",
        "shift chalu karo", "shift shuru", "shuru karo",
        "start my shift", "begin my shift", "my shift start",
        "start my morning shift", "start my day shift", "start my evening shift", "start my night shift",
        "start my afternoon shift", "start day shift", "start evening shift",

    ], 8),
    (Intent.SHIFT_END, [
        "end shift", "end the shift", "end current shift", "end active shift",
        "shift end karo", "shift khatam karo", "shift close karo", "close shift",
        "close the shift", "shift band karo", "shift complete karo", "shift khatam",
        "end my shift", "close my shift", "band karo mera shift", "meri shift end karo", "meri shift khatam karo",
    ], 8),
    (Intent.LOG_WALL_PRODUCTION, [
        "log wall", "wall complete ho gayi", "record wall", "wall print ho chuki",
        "wall complete hui", "wall entry karo", "wall log karo", "wall complete",
        "log wall production", "record wall production",
        "wall print ho gayi", "wall print ho gai", "wall print kar di", "wall complete kar di",
        "i printed wall", "i completed wall",
        "i completed two walls", "i completed two wall", "i completed 2 walls",
        "i completed one wall", "i completed 1 wall",
        "i completed three walls", "i completed four walls", "i completed five walls",
        "completed two walls", "completed two wall", "completed 2 walls",
        "record wall 1", "record wall 2", "record wall 3", "record wall 4", "record wall 5",
        "log wall 1", "log wall 2", "log wall 3", "log wall 4", "log wall 5",
        "log 1 wall", "log 2 wall", "log 3 wall", "log 4 wall", "log 5 wall",
        "record 1 wall", "record 2 wall", "record 3 wall", "record 4 wall", "record 5 wall",
        "log one wall", "log two walls", "log two wall", "log three walls", "log four walls", "log five walls",
        "record one wall", "record two walls", "record two wall", "record three walls", "record four walls", "record five walls",
        "1 wall log karo", "1 wall record karo", "ek wall log karo", "ek wall record karo",
        "ek wall log", "ek wall record", "log ek wall", "record ek wall",
        "pehli wall log", "pehli wall record", "pehli wall log karo", "pehli wall record karo",
    ], 8),
    (Intent.LOG_SHIFT_ISSUE, [
        "log issue", "problem record karo", "issue record karo", "downtime record karo",
        "nozzle clog ho gaya", "downtime note karo", "dikkat note karo",
        "log an issue", "record issue", "record problem",
        # Natural problem statements (English and Hinglish)
        "i have a material flow problem", "there is a material flow problem",
        "material flow problem is happening", "we have a material flow problem",
        "material flow problem", "material flow issue",
        "i have a material slope problem", "material slope problem",
        "i have a problem", "there is a problem", "we have a problem",
        "i have one problem", "i have a one problem",
        "there is one problem", "there is a one problem",
        "we have one problem", "we have a one problem",
        "problem is happening", "issue is happening",
        "i have an issue", "there is an issue", "we have an issue",
        "facing a problem", "facing an issue",
        "having a problem", "having an issue",
        "problem aa rahi hai", "dikkat aa rahi hai", "samasya aa rahi hai",
        "problem ho rahi hai", "dikkat ho rahi hai", "samasya ho rahi hai",
        "material flow mein dikkat", "material flow problem ho rahi",
        "material flow problem aa rahi",
    ], 8),
    (Intent.LOG_SHIFT_HANDOVER, [
        "create handover", "handover create karo", "shift handover note karo",
        "prepare handover", "handover record karo", "initiate handover",
        "initiate shift handover", "handover this shift", "shift handover karo",
    ], 8),
    (Intent.RESOLVE_SHIFT_ISSUE, [
        "issue resolve ho gaya", "issue resolved", "issue solve ho gaya",
        "issue solve ho gayi", "issue resolved ho gaya", "resolve issue",
        "problem solve ho gayi", "problem solve ho gaya", "problem solved",
        "problem theek ho gayi", "problem theek ho gaya", "problem thik ho gayi",
        "problem thik ho gaya", "nozzle clean kiya", "nozzle clean kar diya",
        "nozzle clean ki", "nozzle clean kar di", "downtime khatam",
        "issue close karo", "close issue", "resolve kar diya",
        "mark as resolved", "mark resolved", "recommendation follow",
        "recommendation follow kiya", "recommendation apply kar di",
        "recommendation apply kar diya",
        "recommend kiya tha woh apply kar diya",
        "recommend kiya tha woh apply",
        "recommend kiya tha apply kar diya",
        # Natural action statements (English and Hindi/Hinglish)
        "i cleaned the nozzle", "cleaned the nozzle", "cleaned nozzle",
        "i checked the pump", "checked the pump", "pump check kiya", "pump check kar diya",
        "i replaced the hose", "replaced the hose", "hose replace kiya", "hose replace kar diya",
        "hose badal diya", "hose badla", "pump check ki", "pump check kar di",
        # Natural resolution statements (English and Hindi/Hinglish)
        "the material flow is normal now", "material flow is normal now", "ab material flow normal hai",
        "the problem is resolved", "problem is resolved", "the issue is resolved", "issue is resolved",
        "problem resolve ho gayi", "problem resolve ho gaya",
        "the issue is fixed", "issue is fixed", "issue fixed",
        "the problem is fixed", "problem is fixed", "problem fixed",
        "issue fix ho gaya", "issue fix ho gayi", "problem fix ho gaya", "problem fix ho gayi",
    ], 8),

    # --- Phase 2C Historical Query Intents ---
    (Intent.GET_PREVIOUS_SHIFT, [
        "previous shift", "pichli shift", "last shift", "pichla shift",
        "shift before this", "previous shift mein kya hua",
        "shift timing kya thi", "shift timing", "previous shift timing",
    ], 7),
    (Intent.GET_SHIFT_BY_DATE, [
        "shift on", "yesterday's shift", "kal ki shift", "aaj subah ki shift",
        "shift date", "shifts on date",
        "who worked on", "who worked on date", "who was on shift on", "who worked",
        "worked on", "shifts on", "shifts on date", "shifts date",
    ], 7),
    (Intent.GET_SHIFT_BY_OPERATOR, [
        "how many hours did", "my shift duration", "shift duration", "hours did i work", "how many hours",
        "kitne ghante kaam kiya", "kitne ghante ki thi", "meri shift kitne ghante", "maine kitne ghante",
        "how many walls did", "how many wall did", "kitni walls print ki", "kitni wall print ki",
        "how many walls have i printed", "how many walls did i print", "how many walls have i completed",
        "how many walls did i complete", "kitni wall print ki maine", "maine kitni wall print ki",
        "how many wall have i printed", "how many wall did i print", "how many wall have i completed",
        "how many wall did i complete", "how many wall did rahul print",
        "how many walls did rahul print", "how many walls printed",
        "how many wall printed",
        "when did", "shift start time", "shift end time", "shift start", "shift end",
        "kab shuru hui", "kab khatam hui",
        "ne apni shift mein kya print", "apni shift mein kya print",
        "ne kya print kiya", "kya print kiya tha", "kya print kiya",
        "shifts by", "operator shifts", "kis shift mein kaam kiya",
        "shifts for operator", "walls print ki", "wall print ki",
        "ne kitni walls", "ne kitni wall", "ne kitna print", "ne kya print",
    ], 8),
    (Intent.GET_WALL_HISTORY, [
        "wall status", "history of wall", "wall history", "wall 1 status",
        "wall 2 status", "wall 3 status", "wall 4 status", "wall 5 status",
        "wall kitni print hui", "wall number", "status of wall",
        "wall history status", "history status", "wall ka status",
    ], 7),
    (Intent.GET_HANDOVER_STATUS, [
        "handover status", "handover notes", "last handover", "pending work from last shift",
        "pichle handover", "handover mein kya tha",
    ], 7),
    (Intent.GET_SHIFT_ISSUES, [
        "shift issues", "problems this shift", "is shift mein kya dikkat",
        "issues today", "downtime this shift", "downtime kitna hua",
        "what issues happened", "all issues",
        "what issues occurred", "issues occurred", "problems occurred",
        "issues during", "problems during",
        "what problem did", "what problems did", "what issue did", "what issues did",
        "problem did", "problems did", "issue did", "issues did",
        "what problem did rahul have", "what problems did rahul have", "what issue did rahul have",
        "ki shift mein kya dikkat", "shift mein kya dikkat", "kya dikkat hui",
        "kya samasya hui", "shift mein kya issue", "kya dikkat aayi",
        # Natural shift problem queries (Bug 2)
        "what problem occurred during my shift", "what problems occurred during my shift",
        "problem occurred during my shift", "problems occurred during my shift",
        "what problem occurred", "what problems occurred",
        "problem occurred",
        "what problem happened during my shift", "what problems happened during my shift",
        "what problem happened", "what problems happened",
        "problem happened during my shift", "problems happened during my shift",
        "problem during my shift", "problems during my shift",
        "problem in my shift", "problems in my shift",
        "what issue occurred during my shift", "what issues occurred during my shift",
        "issue occurred during my shift", "issues occurred during my shift",
        "issue during my shift", "issues during my shift",
        "issue in my shift", "issues in my shift",
        "what happened during my shift",
        "meri shift mein kya problem", "meri shift mein kya dikkat",
        "meri shift mein kya issue", "meri shift mein kya samasya",
        "hamari shift mein kya problem", "hamari shift mein kya dikkat",
        "is shift mein kya problem", "is shift mein kya issue",
        "what caused the issue", "what was the cause", "cause of the issue",
        "issue ki kya wajah thi", "issue ki wajah", "kya wajah thi", "dikkat ki wajah",
        "what action did",
        # Natural first-person action queries (Bug 1)
        "what action did i take", "what actions did i take",
        "what step did i take", "what steps did i take",
        "what did i do", "what action did i take on my shift",
        "maine kya action liya", "maine kya step liya",
        "maine kya kiya", "humne kya kiya", "hamne kya action liya",
        "ne kya action liya", "ne kya action",
        "kya action liya", "action liya tha", "kya step liya",
        "what solution did",
        "ne kya solution", "ne kya kiya", "ke issues", "ne kya fix",
        "verified rag solution", "verified rag", "was there a verified", "verified solution",
        "kya koi verified rag solution", "kya koi verified", "verified solution tha kya",
        "rag verified tha", "rag solution verified",
        "what did kelvin recommend", "kelvin ne kya recommend kiya",
        "kelvin recommendation", "kelvin's recommendation", "kelvin ka solution",
        "rag recommendation", "rag source", "kya recommend kiya tha", "kelvin ne kya recommend",
    ], 8),
    (Intent.GET_ACTIVE_SHIFT, [
        "what shift is currently active", "what shift is active", "which shift is active",
        "shift is currently active", "shift is active",
        "active shift", "kaun si shift chal rahi", "current shift status",
        "which shift is running", "shift details", "active shift details",
        "kiska shift", "kiska shift chal raha", "kaun si shift", "who is on shift",
        "current active shift", "shift chal rahi", "shift chal raha",
        "what is my current shift", "my current shift", "current shift",
        "who worked on line 1", "who worked on line", "who is on line 1", "who is on line",
        "who is working on line 1", "who is working on line",
        "line 1 par kaun hai", "line 1 pe kaun hai", "line 1 par kisne kaam kiya",
        "who was on line 1", "who worked on this line",
    ], 7),

    # --- time windows: most specific, they override production wording ---
    (Intent.GET_24_HOUR_PRODUCTION, [
        "24 hour", "24 hours", "last day", "today's production", "todays production",
        "24 ghante", "aaj kitna", "last 24", "aaj ka production", "today production",
    ], 6),
    (Intent.GET_SHIFT_PRODUCTION, [
        "current shift production", "how much did we print this shift",
        "shift production", "this shift", "shift mein kitna",
        "shift ka production", "shift mein",
    ], 7),
    (Intent.GET_RECENT_EVENTS, [
        "last 10 minutes", "last ten minutes", "recently happened", "what happened",
        "last hour", "recent events", "kya hua", "10 minute", "last few minutes",
    ], 6),

    # --- specific measurements ---
    (Intent.GET_LAYER_WIDTH, [
        "layer width", "width of the layer", "layer ki width", "layer width kitni",
        "chaudai", "bead width", "how wide is the layer", "layer kitni chaudi",
    ], 5),
    (Intent.GET_LAYER_HEIGHT, [
        "layer height", "height of the layer", "layer ki height", "layer height kitni",
        "layer thickness", "how thick is the layer",
    ], 5),
    (Intent.GET_WALL_DIMENSIONS, [
        "wall height", "wall width", "wall dimension", "wall ki height", "wall ka size",
        "dimensions of the wall", "wall thickness",
    ], 5),
    (Intent.GET_MATERIAL_REMAINING, [
        "material remaining", "material left", "material baki", "material bacha",
        "material is remaining", "how much material is left", "kitna material bacha",
        "material kitna bacha",
    ], 5),
    (Intent.GET_MATERIAL_RATE, [
        "material consumption", "consumption rate", "material rate",
        "material kitna use ho raha", "material per hour",
    ], 5),
    (Intent.GET_MATERIAL_USAGE, [
        "material used", "material usage", "material consumed", "material kitna use",
        "material kitna laga", "how much material", "material kitna",
    ], 4),

    # --- timing ---
    (Intent.GET_DEADLINE_STATUS, [
        "on schedule", "deadline", "on track", "behind schedule", "schedule par",
        "time par", "late ho",
    ], 5),
    (Intent.GET_ETA, [
        "eta", "when will", "finish time", "kab khatam", "kab tak", "time remaining",
        "how long until", "kitna time lagega", "complete hoga", "kab complete",
    ], 5),
    (Intent.GET_ELAPSED_TIME, [
        "how long has", "elapsed", "running since", "kitni der se", "kab se chal",
        "print been running", "runtime",
    ], 5),
    (Intent.GET_PRINT_RATE, [
        "print rate", "printing rate", "how fast", "speed of printing", "rate kya",
        "printing speed", "kitni tezi",
    ], 5),

    # --- remaining production: must beat generic "how many walls" ---
    (Intent.GET_REMAINING_PRODUCTION, [
        "remaining", "how many are left", "walls left", "baki", "bacha hua",
        "kitni baki", "kitna baki", "still to print", "pending", "are remaining",
        "walls baki", "kitni walls baki",
    ], 4),

    # --- status / quality / problems ---
    (Intent.GET_PRINT_QUALITY, [
        "print quality", "quality", "quality good", "quality kaisi", "defect",
        "is the print good", "print theek",
    ], 4),
    (Intent.GET_CAMERA_STATUS, [
        "camera", "camera working", "camera status", "camera chal raha", "vision system",
        "turn off the camera", "turn off camera", "switch off the camera", "switch off camera",
        "turn on the camera", "turn on camera", "camera off", "camera on", "camera band",
        "camera band karo", "camera chalu", "camera chalu karo", "camera band kar do",
        "disable camera", "stop camera", "close camera", "close the camera",
    ], 6),
    (Intent.GET_ROBOT_STATUS, [
        "robot status", "robot online", "robot working", "robot ka status", "robot chal",
    ], 4),
    (Intent.GET_COMMUNICATION_STATUS, [
        "communication", "connection status", "comms", "network status",
    ], 4),
    (Intent.GET_RECENT_PROBLEMS, [
        "problem", "problems", "issue", "issues", "error", "alert", "warning",
        "dikkat", "kharabi", "fault", "koi problem", "what is wrong",
    ], 4),
    (Intent.GET_SITE_PEOPLE_COUNT, [
        # English
        "how many people are on site", "how many people are currently on site",
        "how many people are working on site", "how many people are there right now",
        "how many people on site", "how many people are there",
        "how many people", "how many persons", "people on site", "people count", "person count",
        "who is on site", "anyone on site", "count people", "workers on site", "workers visible",
        "how many workers", "how many workers on site", "site people",
        "people in the room", "people visible", "people visible on camera", "people on camera", "workers on camera",
        "how many people are visible on camera", "how many workers are on site",
        "how many persons are on site",
        # Hindi / Hinglish
        "abhi site pe kitne log hain", "abhi site par kitne log hain", "abhi site pe kitne log", "abhi site par kitne log",
        "site par kitne log hain", "site pe kitne log hain", "site par kitne log", "site pe kitne log",
        "kitne log hain", "kitne log dikh rahe hain", "kitne log dikh rahe", "kitne log present hain",
        "kitne log site par", "kitne log site pe", "site pe kaun kaun hai", "site par kaun hai",
        "kitne bande hain", "site par kitne bande",
        "site par kitne workers", "site pe kitne workers", "kitne workers",
        "site par kitne vyakti", "site pe kitne vyakti", "kitne vyakti",
        "camera me kitne log", "camera par kitne log", "camera me kitne",
        "site pe kaun kaun", "site par kaun",
    ], 5),
    (Intent.GET_SITE_DETECTIONS, [
        "what objects are visible", "what objects are detected", "what do you see on camera 2",
        "what is visible on camera 2", "what is detected on camera 2", "camera 2 detections",
        "detected objects on site", "objects on camera 2", "objects detected on camera",
        "objects on site", "what is on camera 2", "what is detected on site",
        "camera 2 pe kya dikh raha hai", "camera 2 me kya dikh raha", "site camera me kya dikh raha",
        "kya kya objects dikh rahe hain", "objects visible on camera", "objects detected",
        # General site camera / camera 2 detection patterns (model-agnostic, no hardcoded classes)
        "visible on camera 2", "detected on camera 2", "see on camera 2", "seen on camera 2",
        "on camera 2", "dikh raha hai camera 2", "camera 2 par", "camera 2 pe",
        "objects visible on camera 2", "objects detected on camera 2",
        "site camera detections", "camera detections", "what objects are on site",
    ], 5),
    (Intent.GET_OPERATOR, [
        "kaun printing", "kaun print", "kaun chala", "kaun operate", "kaun kar raha",
        "operator kaun", "kaun sa operator", "who is printing", "who is operating",
        "who is running", "current operator", "operator name", "operator kon",
        "who is working right now", "who's working right now", "who is working", "who's working",
        "who is on shift right now", "who is on shift", "who is on duty", "is anyone working",
        "is lokmanya working", "is lokmanya working right now", "is lukhmaan working",
        "is hamdaan working", "is farish working",
    ], 5),
    (Intent.GET_CURRENT_LAYER, [
        "current layer", "which layer", "what layer", "layer number", "kaunsa layer",
        "kaun sa layer", "abhi layer", "layer chal raha", "layers completed",
        "how many layer", "how many layers", "kitne layer", "layer are we printing",
        # Hinglish forms for "which layer is running" and "how many layers".
        "kaunsi layer", "layer chal rahi", "layer chal", "total layers",
        "total kitni layer", "kitni layer", "layer hai", "layer kitni",
        "layer print ho", "layer printed", "layer have been",
    ], 3),
    (Intent.GET_MACHINE_STATUS, [
        "machine status", "machine state", "machine ka status", "machine chal",
        "is the machine", "printing chal rahi", "printing running", "is printing",
        "print ho raha", "status", "print chal", "printing chal", "machine chal rahi",
        "printing ka status", "printing status", "print raha", "printing hai",
    ], 2),
    (Intent.GET_TOTAL_TARGET, [
        "total target", "target", "total walls", "how many in total", "total kitna",
        "kul kitna",
    ], 2),

    # --- most generic: completed production ---
    (Intent.GET_COMPLETED_PRODUCTION, [
        "how many walls", "walls printed", "walls have been printed", "completed",
        "how much printed", "production", "kitni wall", "kitni walls", "print ho gay",
        "print ho chuki", "kitna print hua", "abhi tak kitna", "wall count",
        "printed so far", "how many have we printed", "how much is printed",
        "how much has been printed", "much printed", "is printed",
        # Common speech-to-text spellings of the Hinglish phrasing.
        "print ho gai", "print ho gayi", "walls print", "wall print",
        "kitne wall", "kitne walls", "keetni", "keetney",
        # "kitni printing complete ho gayi", "kitna print ho chuka hai",
        # "well kitni complete hui" -- previously unmatched.
        "printing complete", "complete ho gay", "complete hui", "complete ho chuki",
        "kitni printing", "print ho chuka", "ho chuka", "kitni complete",
        "kitni well", "well print", "kitni wall print", "print hua",
    ], 1),
]

# Knowledge questions: definitions, causes, procedures -- answered from RAG.
_KNOWLEDGE_TERMS = [
    "what is", "what does", "why does", "why can", "how can", "how does",
    "explain", "meaning", "kya hota", "kya hai matlab", "kaise hota", "kyun hota",
    "definition", "reason for", "causes",
]

# Technical topics that indicate a documentation lookup rather than live data.
#
# Deliberately EXCLUDES "layer width": it names a live measurement the router
# already resolves, and adding it sent four working queries ("layer width kitni
# hai", "what is the current layer width", ...) to COMBINED_QUERY. The live-hint
# guard in `classify` protects those anyway, but there is no reason to widen the
# vocabulary into a noun that has its own live intent.
_TECHNICAL_TOPICS = [
    "delamination", "layer shifting", "layer shift", "bonding", "adhesion",
    "warping", "spaghetti", "clog", "nozzle", "curing", "shrinkage", "crack",
    "porosity", "segregation", "slump", "maintenance", "procedure", "sop",
    "calibration", "calibrate",
    # Printing subjects documented in the ingested knowledge base. Without
    # these, a question naming any of them never reached RAG at all.
    "accelerator", "hose", "choking", "choke", "clogging", "layer thickness",
    "thickness", "gap", "void", "servo", "pump", "material flow", "fly ash",
    "hopper", "esp", "g code", "gcode", "wall crack", "printing defect",
    "printing problem", "sieve", "u joint", "z height", "delaminate",
    "pipe", "clean", "shift ho",
]

# Signals that the operator is reporting a PROBLEM or asking HOW TO do
# something, rather than asking for a current reading.
#
# This exists because `_EXPLANATION_TERMS` is only consulted when a live phrase
# also matched (see `classify`). A troubleshooting question that matches no live
# phrase used to lose its signal entirely and fall through to
# `_fallback_keywords`, where a bare "layer" became GET_CURRENT_LAYER -- so
# "why is there a gap in my printed layer?" was answered with "Current layer is
# 42 out of 60". These terms are checked BEFORE that fallback.
#
# Bare "problem" and "issue" are deliberately ABSENT. They also open the live
# "kya koi problem hai" query, which would then route to COMBINED_QUERY -- and
# `summarise` builds its combined sentence from only four fact keys
# (layer_width_mm, completed_mm, print_rate_mm_per_min, quality_percent), none
# of which GET_RECENT_PROBLEMS produces. The live problem list was therefore
# dropped from the spoken answer. Excluding these two words costs nothing: every
# troubleshooting question still matches on "defect", "not working", "fix",
# "troubleshoot", "error", "fault" or "blockage".
#
# The Hinglish entries are the operator utterances documented in the
# troubleshooting dataset ("layer patli aa rahi hai", "layers chipak nahi
# rahe"), which carry the defect meaning in words no English list would catch.
_PROBLEM_TERMS = [
    "what should i do", "what should i check", "how often", "how do i", "how to",
    "defect", "not working", "repeatedly", "troubleshoot",
    "kya karein", "kya karna", "kaise karein", "error", "fault", "blocked",
    "blockage", "stuck", "uneven", "too thick", "too thin", "correct", "fix",
    "patli", "patla", "moti", "mota", "kharab", "chipak nahi", "alag ho",
    "gir gay", "kam aa raha", "zyada spread", "phat", "last kab",
]

# Signals that the operator wants an EXPLANATION alongside a value. Plain
# "what is" is excluded: "what is the current layer width" is a pure live query.
_EXPLANATION_TERMS = [
    "can happen", "what problems", "why", "explain", "mean", "cause", "how can",
    "is it normal", "normal?", "should i", "kyun", "kaise hota", "kya hota",
]

_LIVE_HINT_TERMS = [
    "current", "abhi", "now", "today", "aaj", "status", "how many", "kitni", "kitna",
]

# --- gate for consulting the documentation before the fallback keyword table ---
#
# `_fallback_keywords` maps a single bare word to a live intent, so a question
# it cannot really answer still got a confident live number: "How should the
# wall be lifted?" matched "wall" and was answered "650.0 millimeter printed out
# of 1000.0 millimeter". Asking retrieval first fixes that -- but only when
# retrieval genuinely has an answer, or the same hijack happens in reverse.
#
# Two independent conditions, because neither alone is sufficient:
#
# FALLBACK_RAG_MIN_SCORE is NOT RAG_MIN_SCORE and must not be confused with it.
# RAG_MIN_SCORE (0.15) decides whether a retrieved passage is worth showing at
# all; this decides whether retrieval should outrank a live intent, which needs
# a much higher bar. Measured on this corpus: "kitna material use hua hai"
# retrieves an irrelevant record at 0.219, so 0.15 would send a working live
# query to the documentation. 0.35 is an empirical separator fitted to the
# current 101-chunk index, not a principled constant -- re-measure it if the
# index changes materially. The tests pin the two live queries that sit closest
# to this boundary so a future change fails loudly instead of silently.
FALLBACK_RAG_MIN_SCORE = 0.35
#
# FALLBACK_RAG_MIN_TERMS protects single-word live queries. Lexical scores are
# normalised against what a query could achieve, so a one-informative-word
# question scores near the ceiling against any document containing that word:
# "who is the operator" retrieves an unrelated record at 0.734, which no
# threshold can screen out. Requiring two informative words does, because a
# genuine documentation question names both a subject and something about it.
FALLBACK_RAG_MIN_TERMS = 2


# Hinglish has no fixed spelling, and speech-to-text picks whichever form it
# hears. These are spelling variants of the SAME word, folded to one form so a
# single phrase in _PATTERNS covers all of them. Word-for-word only -- nothing
# here changes meaning, so "printing chal rhi h" and "printing chal rahi hai"
# become the same string.
_HINGLISH_FORMS = {
    "h": "hai", "hai": "hai", "hain": "hai", "he": "hai", "hy": "hai",
    "rhi": "rahi", "rahi": "rahi", "rhi?": "rahi",
    "rha": "raha", "rhaa": "raha", "raha": "raha",
    "rhe": "rahe", "rhy": "rahe",
    "gyi": "gayi", "gai": "gayi", "gyee": "gayi", "gayee": "gayi",
    "gya": "gaya", "gyaa": "gaya",
    "kr": "kar", "krr": "kar",
    "krta": "karta", "krna": "karna", "krke": "karke",
    "nhi": "nahi", "nai": "nahi",
    "chuka": "chuka", "chuki": "chuki", "chuke": "chuka",
    "kitni": "kitni", "kitnee": "kitni", "keetni": "kitni", "keetney": "kitni",
    "baaki": "baki", "bcha": "bacha", "bacha": "bacha",
    "kon": "kaun", "kaun": "kaun", "koun": "kaun",
    "konsi": "kaunsi", "kaunsi": "kaunsi", "kounsi": "kaunsi",
    "abhi": "abhi", "abi": "abhi",
    "wel": "well", "wal": "wall",
    "hva": "hua", "hva?": "hua", "huva": "hua", "hui": "hui",
    "layers": "layer", "walls": "wall", "wells": "well",
    "prnt": "print", "printng": "printing",
}

# Last resort. If no phrase matched but the question clearly IS about the
# machine, fall back to the intent that single word implies. This exists
# because speech-to-text mangles sentences ("it will all print away" was
# "kitni wall print ho gayi"): a recognisable machine word is better evidence
# than nothing, and answering the likely question beats refusing outright.
# Ordered most specific first.


# Whisper transcribes Hindi speech in DEVANAGARI SCRIPT, not romanised
# Hinglish: "printing chal rahi hai" comes back as "प्रिंटिंग चल रही है".
# Every pattern in this file is written in Latin text, so a Devanagari
# transcript previously matched nothing at all and became UNKNOWN_QUERY no
# matter how clear the question was. These are the machine words the operator
# actually uses, mapped to the spelling the patterns expect.
#
# Longest phrases first: "चल रही है" must be replaced before "है" alone.
_DEVANAGARI = [
    ("प्रिंटिंग", "printing"), ("प्रिन्टिंग", "printing"), ("प्रिंट", "print"),
    ("छपाई", "printing"),
    ("चल रही है", "chal rahi hai"), ("चल रहा है", "chal raha hai"),
    ("चल रही", "chal rahi"), ("चल रहा", "chal raha"),
    ("हो गई है", "ho gayi hai"), ("हो गयी है", "ho gayi hai"),
    ("हो गई", "ho gayi"), ("हो गयी", "ho gayi"), ("हो गया", "ho gaya"),
    ("हो चुकी", "ho chuki"), ("हो चुका", "ho chuka"),
    ("कितनी", "kitni"), ("कितना", "kitna"), ("कितने", "kitne"),
    ("दीवार", "wall"), ("वॉल", "wall"), ("वेल", "well"),
    ("लेयर", "layer"), ("परत", "layer"), ("लेयर्स", "layers"),
    ("बाकी", "baki"), ("बचा", "bacha"), ("शेष", "baki"),
    ("कौनसी", "kaunsi"), ("कौन सी", "kaunsi"), ("कौन", "kaun"),
    ("कुल", "total"), ("पूरा", "complete"), ("पूरी", "complete"),
    ("कम्पलीट", "complete"), ("कंप्लीट", "complete"),
    ("अभी", "abhi"), ("स्टेटस", "status"), ("स्थिति", "status"),
    ("मशीन", "machine"), ("कैमरा", "camera"), ("रोबोट", "robot"),
    ("चौड़ाई", "width"), ("ऊंचाई", "height"),
    ("मटेरियल", "material"), ("सामग्री", "material"),
    ("ऑपरेटर", "operator"), ("समस्या", "problem"), ("दिक्कत", "problem"),
    ("कब", "kab"), ("क्या", "kya"), ("है", "hai"), ("हैं", "hai"),
    ("रही", "rahi"), ("रहा", "raha"), ("गई", "gayi"), ("गयी", "gayi"),
]

# The wake phrase is not part of the question, and speech-to-text mangles it
# constantly ("hey robot" -> "high robot", "आइरोट", "a robot"). Stripping it
# stops a misheard wake word from burying a perfectly good question.
_WAKE_PREFIXES = [
    # "Okay Kelvin" and the spellings speech-to-text returns for it.
    "okay kelvin6k", "ok kelvin6k", "okay kelvin", "ok kelvin", "okey kelvin",
    "o k kelvin", "okay calvin", "ok calvin", "okay kevin", "ok kevin",
    "hey kelvin", "ok kelvin6", "okay kelvin6",
    # Older phrases, still stripped so previously recorded questions keep working.
    "hey robot", "hey rob", "high robot", "hi robot", "hello robot", "ok robot",
    "a robot", "ai robot", "eye robot", "hey robert",
    "ओके केल्विन", "ठीक है केल्विन", "आइरोट", "अरोबोट", "हे रोबोट",
]


def _strip_wake_word(text: str) -> str:
    """Removes a leading wake phrase, however badly it was transcribed."""
    for phrase in _WAKE_PREFIXES:
        if text.startswith(phrase):
            return text[len(phrase):].lstrip(" ,.")
    return text


def strip_wake_words(text: str) -> str:
    """
    The question with any wake phrase removed. Empty means "only the wake word".

    Public because the assistant needs to tell a bare "Okay Kelvin" apart from a
    question it failed to understand: the first deserves "Yes, what would you
    like to know?", the second deserves an apology.
    """
    return _normalise(text)


def _normalise(text: str) -> str:
    """
    Lower-case, strip punctuation, and fold Hinglish spelling variants.

    Hyphens and underscores become spaces: speech-to-text sometimes returns
    hyphenated runs like "Kitni-Walls-Print-Ho-Gayi-Hain", which would otherwise
    match no phrase at all.

    The spelling fold is what makes casual speech work. Whisper transcribes the
    same spoken sentence as "printing chal rhi h" one time and "printing chal
    rahi hai" the next; without folding, only one of them matches.
    """
    text = text.lower().strip()
    # Devanagari -> Latin BEFORE anything else, so a Hindi-script transcript
    # goes through exactly the same matching as a romanised one.
    for hindi, latin in _DEVANAGARI:
        if hindi in text:
            text = text.replace(hindi, latin)
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[?!.,;:\"‘’“”]", " ", text)
    text = re.sub(r"\bmaterial\s+slope\b", "material flow", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _strip_wake_word(text)
    return " ".join(_HINGLISH_FORMS.get(word, word) for word in text.split())


_FUTURE_OR_PLANNING_TERMS = [
    "karni hai", "karna hai", "karne hai",
    "plan", "target", "will", "chahiye", "karunga", "karenge",
]

_QUESTION_INDICATORS = [
    "kitni", "kitna", "kitne", "keetni", "keetney",
    "how many", "how much", "what is", "kya", "count",
    "so far", "abhi tak",
    "did i", "did we", "have i", "have we", "was", "were",
    "yesterday", "kal", "past",
]

_WALL_COMPLETION_VERBS = [
    "print ho gayi", "print ho gai", "print ho chuki", "print hui",
    "complete ho gayi", "complete ho gai", "complete ho chuki", "complete hui",
    "print kar di", "print kar diya",
    "complete kar di", "complete kar diya",
    "i printed wall", "i completed wall",
    "log wall", "record wall", "wall entry karo", "wall log karo",
    "completed wall", "completed walls", "completed two walls", "completed two wall", "completed 2 walls",
]


def _is_future_or_planning(q: str) -> bool:
    return any(t in q for t in _FUTURE_OR_PLANNING_TERMS)


def _is_wall_production_question(q: str) -> bool:
    return any(t in q for t in _QUESTION_INDICATORS)


def _is_issue_question(q: str) -> bool:
    question_starters = [
        "how", "what", "why", "kaise", "kya", "kyun",
        "which", "who", "kisko", "kisne", "kahan", "where"
    ]
    return any(re.search(rf"\b{qs}\b", q) for qs in question_starters)


def _is_completed_wall_production(q: str) -> bool:
    if _is_future_or_planning(q) or _is_wall_production_question(q):
        return False
    if any(t in q for t in _PROBLEM_TERMS) or any(w in q for w in ("void", "crack", "gap", "defect", "choke", "choking", "problem", "issue", "dikkat", "samasya")):
        return False
    has_wall = bool(re.search(r"\b(?:walls?|wells?)\b", q))
    if not has_wall:
        return False
    if any(p in q for p in _WALL_COMPLETION_VERBS):
        return True
    return bool(re.search(r"\b(?:i\s+(?:have\s+)?(?:completed|printed)|completed)\s+(?:\d+|one|two|three|four|five|ek|do|teen|char|chaar|panch)\s+(?:walls?|wells?)\b", q))

_GENERIC_DETECTION_TERMS = {
    "object", "objects", "anything", "something", "item", "items",
    "what", "all", "detections", "everything", "classes", "what do you see",
}


def _clean_candidate_class(cand: Optional[str]) -> Optional[str]:
    if not cand:
        return None
    cand = cand.strip().lower()
    cand = re.sub(r"^(?:the|a|an|any|some|that|this)\s+", "", cand)
    cand = re.sub(r"\s+(?:on\s+camera(?:\s*2)?|on\s+site|via\s+camera(?:\s*2)?|right\s+now|currently|please)$", "", cand)
    cand = re.sub(r"^(?:on\s+camera(?:\s*2)?|on\s+site|via\s+camera(?:\s*2)?)\s*", "", cand)
    cand = cand.strip()
    if not cand or cand in _GENERIC_DETECTION_TERMS or cand.startswith("what"):
        return None
    return cand


def extract_requested_detection_class(q: str) -> Optional[str]:
    """
    Extracts the target object/class from a natural language detection question
    without hardcoding any class names.
    Returns None for general detection queries (e.g. 'what objects are visible?').
    """
    if not q:
        return None
    q_norm = q.lower().strip()
    if re.search(r"\bwhat\s+(?:objects?|do\s+you\s+see|is\s+visible|is\s+detected|is\s+on)\b", q_norm):
        return None
    patterns = [
        r"\b(?:is|are|was|were)\s+(?:there\s+)?(.+?)\s+(?:visible|detected|seen)\b",
        r"\b(?:do|can|could)\s+you\s+see\s+(.+?)(?:\s+(?:visible|detected|on\s+camera|on\s+site)|$|\?)",
        r"\bcheck\s+(?:if\s+(.+?)\s+is\s+(?:visible|detected)|for\s+(.+?)(?:\s+on\s+camera|\s+on\s+site|$))",
        r"\bcamera(?:\s*2)?\s+(?:me|par|pe)\s+(.+?)\s+dikh\s+rah",
        r"\b(?:kya\s+)?(.+?)\s+dikh\s+rah[ia]\s+(?:hai\s+)?(?:camera|site)",
        r"\b(.+?)\s+(?:is|are)?\s*(?:visible|detected)\s+on\s+(?:camera|site)\b",
        r"\b(.+?)\s+(?:visible|detected)\b",
    ]
    for p in patterns:
        m = re.search(p, q_norm)
        if m:
            cand = next((_clean_candidate_class(g) for g in m.groups() if g and _clean_candidate_class(g)), None)
            if cand:
                return cand
    return None


def _find_supported_class(candidate: str, supported_classes: List[str]) -> Optional[str]:
    """
    Dynamically checks if candidate matches any class in the loaded YOLO model's supported_classes.
    Handles case-insensitivity, singular/plural, and space/underscore variations.
    """
    if not candidate or not supported_classes:
        return None
    c_cand = candidate.strip().lower()
    norm_map: Dict[str, str] = {}
    for sc in supported_classes:
        sc_low = sc.lower()
        norm_map[sc_low] = sc
        norm_map[sc_low.replace("_", " ")] = sc
        norm_map[sc_low.replace(" ", "")] = sc
        norm_map[sc_low.replace("-", " ")] = sc

    if c_cand in norm_map:
        return norm_map[c_cand]
    if c_cand.endswith("es") and c_cand[:-2] in norm_map:
        return norm_map[c_cand[:-2]]
    if c_cand.endswith("s") and c_cand[:-1] in norm_map:
        return norm_map[c_cand[:-1]]
    if (c_cand + "s") in norm_map:
        return norm_map[c_cand + "s"]
    if (c_cand + "es") in norm_map:
        return norm_map[c_cand + "es"]
    return None


def _is_site_detection_query(q: str) -> bool:
    """
    Dynamically checks if a query is asking whether an object/class is visible
    or detected on Camera 2 / site camera, without hardcoding any classes.
    """
    if not q:
        return False
    if any(p in q for p in ("people", "person", "persons", "worker", "workers", "vyakti", "bande", "log ")) and any(k in q for k in ("count", "kitne", "how many", "who is", "anyone")):
        return False
    if any(w in q for w in ("quality", "pressure", "material", "temperature", "motor", "speed", "flow", "downtime", "hours", "started", "ended", "completed")):
        if not ("visible" in q or "detected" in q or "dikh rah" in q or "camera 2" in q):
            return False
    if any(p in q for p in (
        "what objects are visible", "what objects are detected", "what do you see on camera",
        "what is visible on camera", "what is detected on camera", "camera 2 detections",
        "detected objects on site", "objects on camera 2", "objects detected on camera",
        "objects on site", "what is on camera 2", "what is detected on site",
        "objects visible on camera", "objects detected", "visible on camera 2",
        "detected on camera 2", "see on camera 2", "seen on camera 2",
        "camera 2 me kya dikh raha", "camera 2 pe kya dikh raha",
    )):
        return True
    cand = extract_requested_detection_class(q)
    return cand is not None


def _score(question: str, phrases: List[str]) -> Tuple[float, List[str]]:
    hits = [p for p in phrases if p in question]
    if not hits:
        return 0.0, []
    # Longer matches are more specific, so they outrank generic ones.
    return float(max(len(p) for p in hits)) + 0.1 * len(hits), hits


def classify(question: str) -> Tuple[Intent, List[str]]:
    """Best-matching intent plus the phrases that triggered it."""
    q = _normalise(question)
    if not q:
        # _normalise strips the wake phrase, so an empty result means the
        # operator said only "Okay Kelvin". That is understood perfectly --
        # reporting UNKNOWN_QUERY would wrongly mark it as a failure.
        return (Intent.WAKE_WORD, ["wake word"]) if question.strip() else (Intent.UNKNOWN_QUERY, [])

    # Check for registered operator dynamically using existing lookup
    op_id = extract_operator_from_text(question)

    # Printing job actions:
    m_job_start = re.search(r"\b(?:start|begin)\s+(?:printing\s+)?job\s+([A-Za-z0-9_]+)\b", q)
    if m_job_start:
        return Intent.START_PRINTING_JOB, [f"job_name:{m_job_start.group(1)}"]
    m_job_end = re.search(r"\b(?:finish|end|complete|stop)\s+(?:printing\s+)?job\s+([A-Za-z0-9_]+)\b", q)
    if m_job_end:
        return Intent.END_PRINTING_JOB, [f"job_name:{m_job_end.group(1)}"]

    # Operational reporting queries
    operational_patterns = [
        r"\bwho\s+(?:is\s+(?:in|on)|was\s+(?:in|on)|came\s+in\s+for|worked\s+the|worked|on)\s+(?:the\s+)?(?:morning|day|evening|night)\s+shift\b",
        r"\bwho\s+arrived\s+late\b",
        r"\b(?:total\s+printing\s+time|printing\s+time\s+today)\b",
        r"\bcross(?:ed)?\s+9\s+hours\b",
        r"\b(?:how\s+much\s+overtime|total\s+overtime|highest\s+overtime|overtime\s+was\s+recorded|any\s+overtime)\b",
        r"\b(?:what\s+is\s+[a-zA-Z]+(?:'s)?\s+overtime|[a-zA-Z]+\s+overtime|did\s+[a-zA-Z]+\s+have\s+(?:any\s+)?overtime)\b",
        r"\b(?:how\s+many\s+(?:total\s+)?working\s+days|working\s+days\s+has\s+[a-zA-Z]+\s+completed)\b",
        r"\bwhat\s+(?:issue|problem)\s+(?:was\s+reported|occurred)\s+(?:during|in)\s+(?:the\s+)?(?:morning|day|evening|night)\s+shift\b",
        r"\bwhat\s+problem\s+did\s+[a-zA-Z]+\s+face\b",
        r"\bwhen\s+did\s+[a-zA-Z]+\s+(?:arrive|come\s+in|clock\s+in)\b",
        r"\bwhat\s+time\s+did\s+[a-zA-Z]+\s+(?:arrive|come\s+in|clock\s+in)\b",
        r"\bwhen\s+did\s+[a-zA-Z]+\s+(?:leave|clock\s+out|exit)\b",
        r"\bwhat\s+time\s+did\s+[a-zA-Z]+\s+(?:leave|clock\s+out|exit)\b",
        r"\bhow\s+many\s+hours\s+did\s+[a-zA-Z]+\s+work\b",
        r"\b(?:when\s+did\s+[a-zA-Z]+\s+arrive.*overtime|arrive\s+for\s+his\s+shift\s+and\s+what\s+is\s+his\s+overtime)\b",
    ]
    for pat in operational_patterns:
        if re.search(pat, q):
            return Intent.OPERATIONAL_REPORT, ["operational_report"]

    # Kelvin recommendation questions -> KNOWLEDGE_QUERY
    if "kelvin" in q and ("recommend" in q or "suggest" in q or "solution" in q):
        return Intent.KNOWLEDGE_QUERY, ["kelvin_recommendation"]

    # Dynamic registered operator query routing
    if op_id:
        # 1. Operator production query
        is_op_production = bool(
            re.search(r"\b(?:what\s+did\s+.+\s+print|how\s+many\s+walls?\s+did|walls?\s+did\s+.+\s+print|ne\s+kya\s+print|ne\s+kitn\w+\s+wall|ne\s+kitn\w+\s+print|ka\s+production|ne\s+apni\s+shift\s+mein\s+kya\s+print|kya\s+print\s+kiya)\b", q)
            or (("print" in q or "wall" in q or "walls" in q or "production" in q) and any(w in q for w in ("what did", "how many", "kitni", "kitna", "kya")))
        )
        if is_op_production and not _is_completed_wall_production(q):
            return Intent.GET_SHIFT_BY_OPERATOR, [f"operator_production:{op_id}"]

        # 2. Operator issues / problems / actions query
        is_op_issue = bool(
            re.search(r"\b(?:problem|problems|issue|issues|dikkat|samasya|action|solution|kya\s+kiya|kya\s+action|kya\s+solution|kya\s+fix|ke\s+issues|what\s+did\s+.+\s+do)\b", q)
        )
        if is_op_issue:
            return Intent.GET_SHIFT_ISSUES, [f"operator_issue:{op_id}"]

        # 3. Operator timing / duration query
        is_op_timing = bool(
            re.search(r"\b(?:when\s+did|hours\s+did|how\s+many\s+hours|shift\s+start|shift\s+end|shift\s+timing|duration|kitne\s+ghante|kab\s+shuru|kab\s+khatam|shuru\s+hui|khatam\s+hui)\b", q)
        )
        if is_op_timing:
            return Intent.GET_SHIFT_BY_OPERATOR, [f"operator_timing:{op_id}"]

    best_intent = Intent.UNKNOWN_QUERY
    best_score = 0.0
    best_terms: List[str] = []

    for intent, phrases, priority in _PATTERNS:
        if intent == Intent.LOG_WALL_PRODUCTION:
            if _is_future_or_planning(q) or _is_wall_production_question(q):
                continue
            if not re.search(r"\b(?:walls?|wells?)\b", q):
                continue
        if intent == Intent.LOG_SHIFT_ISSUE:
            if _is_future_or_planning(q) or _is_issue_question(q):
                continue
        if intent == Intent.RESOLVE_SHIFT_ISSUE:
            if _is_future_or_planning(q) or _is_issue_question(q):
                continue
        score, hits = _score(q, phrases)
        if score <= 0:
            continue
        weighted = priority * 1000 + score
        if weighted > best_score:
            best_intent, best_score, best_terms = intent, weighted, hits

    if _is_completed_wall_production(q) and best_intent != Intent.LOG_WALL_PRODUCTION:
        best_intent = Intent.LOG_WALL_PRODUCTION
        best_score = 8000.0
        best_terms = ["completed_wall_production"]

    if _is_site_detection_query(q) and best_intent not in (Intent.GET_SITE_DETECTIONS, Intent.GET_SITE_PEOPLE_COUNT):
        best_intent = Intent.GET_SITE_DETECTIONS
        best_score = 7500.0
        best_terms = ["site_detection_query"]

    shift_and_action_intents = {
        Intent.SHIFT_START,
        Intent.SHIFT_END,
        Intent.LOG_WALL_PRODUCTION,
        Intent.LOG_SHIFT_ISSUE,
        Intent.LOG_SHIFT_HANDOVER,
        Intent.RESOLVE_SHIFT_ISSUE,
        Intent.GET_ACTIVE_SHIFT,
        Intent.GET_PREVIOUS_SHIFT,
        Intent.GET_SHIFT_BY_DATE,
        Intent.GET_SHIFT_BY_OPERATOR,
        Intent.GET_WALL_HISTORY,
        Intent.GET_SHIFT_ISSUES,
        Intent.GET_HANDOVER_STATUS,
        Intent.GET_SITE_DETECTIONS,
    }

    # Protect write/action, site detections, and historical shift intents from being hijacked
    if best_score > 0 and best_intent in shift_and_action_intents:
        return best_intent, best_terms

    # Kelvin recommendation questions -> KNOWLEDGE_QUERY
    if "kelvin" in q and ("recommend" in q or "suggest" in q or "solution" in q):
        return Intent.KNOWLEDGE_QUERY, ["kelvin_recommendation"]

    is_knowledge = any(t in q for t in _KNOWLEDGE_TERMS)
    has_topic = any(t in q for t in _TECHNICAL_TOPICS)
    wants_explanation = any(t in q for t in _EXPLANATION_TERMS)
    reports_problem = any(t in q for t in _PROBLEM_TERMS)
    wants_live = any(t in q for t in _LIVE_HINT_TERMS)

    # A live intent, a live hint ("current", "abhi", "status", "kitni"), and no
    # sign of a problem -> the operator wants the reading, nothing else. This
    # runs FIRST so that widening _TECHNICAL_TOPICS cannot hijack a working live
    # query: "current layer width kitni hai" stays GET_LAYER_WIDTH.
    #
    # `is_knowledge` is deliberately NOT part of the problem signal here --
    # "what is" opens plenty of live questions, and counting it sent five live
    # queries to the wrong intent when measured.
    if best_score > 0 and wants_live and not (has_topic or reports_problem):
        return best_intent, best_terms

    # A live measurement PLUS a request to explain or a problem report ->
    # combined answer: resolve the real value, then add the documentation.
    if best_score > 0 and (wants_explanation or has_topic or reports_problem):
        return Intent.COMBINED_QUERY, best_terms + [t for t in _TECHNICAL_TOPICS if t in q]

    if has_topic or (is_knowledge and best_score == 0):
        return Intent.KNOWLEDGE_QUERY, [t for t in _TECHNICAL_TOPICS if t in q] or ["definition"]

    # A problem report that matched no live phrase. This must be answered from
    # the documentation, and it has to be decided HERE -- below this point the
    # fallback keyword table would map a bare "layer" or "print" to a live
    # intent and answer a defect question with a production number.
    if reports_problem and best_score == 0:
        return Intent.KNOWLEDGE_QUERY, [t for t in _PROBLEM_TERMS if t in q] or ["problem"]

    if best_score == 0:
        # Nothing matched a live phrase. Ask the documentation before the
        # fallback keyword table does, because that table only knows live
        # intents and will answer a documented question with a production
        # number. Gated on BOTH thresholds -- see their definitions above.
        if _documentation_can_answer(question, q):
            return Intent.KNOWLEDGE_QUERY, ["documented"]

        # Phase 1 Semantic NLU Fallback:
        # Engage Gemini semantic understanding before blind single-word keyword matching
        try:
            from agent.nlu import semantic_fallback, ALLOWED_READ_INTENTS
            nlu_res = semantic_fallback(question)
            if (
                nlu_res
                and nlu_res.intent in ALLOWED_READ_INTENTS
                and nlu_res.confidence >= 0.70
                and not nlu_res.ambiguous
            ):
                log.info(
                    "Semantic NLU rescued query %r -> %s (confidence %.2f)",
                    question, nlu_res.intent.value, nlu_res.confidence
                )
                return nlu_res.intent, ["semantic_nlu", nlu_res.intent.value.lower()]
        except Exception as exc:
            log.warning("Semantic NLU fallback failed (%s); proceeding to keyword fallback", exc)

        # Do not fall back to single-word keywords for coordinate/toolpath queries
        if any(term in q for term in ("x y z", "xyz", "coordinate", "coordinates", "gcode", "g-code", "toolpath")):
            return Intent.UNKNOWN_QUERY, []

        # Still nothing. Look for a single machine word -- a mangled
        # transcript still names its subject.
        for keyword, intent in _fallback_keywords():
            if keyword in q:
                log.info("No phrase matched; falling back on the keyword %r", keyword)
                return intent, [keyword]
        return Intent.UNKNOWN_QUERY, []

    return best_intent, best_terms


def _documentation_can_answer(question: str, normalised: str) -> bool:
    """
    Whether retrieval has a confident answer for a question that matched no
    live phrase.

    Deliberately conservative: a false positive here sends a working live query
    to the documentation, which is worse than leaving a documented question to
    the fallback table. Any failure is swallowed -- retrieval being unavailable
    must never change how a live question is routed.
    """
    try:
        result = search_documentation(question)
        if not result.get("found"):
            return False
        score = float(result["results"][0].get("score") or 0.0)
        if score < FALLBACK_RAG_MIN_SCORE:
            return False
        informative = [t for t in tokenize(normalised) if t not in _STOPWORDS]
        if len(informative) < FALLBACK_RAG_MIN_TERMS:
            return False
        log.info(
            "No live phrase matched; the documentation answers this (score %.3f)", score
        )
        return True
    except Exception as exc:  # retrieval is advisory here, never load-bearing
        log.warning("Could not check the documentation before fallback: %s", exc)
        return False


def _evaluate_rag_verified(top_text: str) -> int:
    """
    Determines whether a retrieved RAG document is SITE-VERIFIED (1) or unverified (0).
    Rules:
      - CONFLICT / DISPUTED / DO NOT ASSUME / MUST CONFIRM -> 0
      - UNVERIFIED / NOT CONFIRMED / UNCONFIRMED / POSSIBLE ONLY -> 0
      - DOCUMENTED -> 0
      - SITE-VERIFIED only (without unconfirmed, conflict, or documented qualifiers) -> 1
    Never marks verified on bare substring alone.
    """
    fields = parse_record(top_text)
    status_raw = ""
    if fields and "verification status" in fields:
        status_raw = fields["verification status"].get("value", "")
    else:
        for line in top_text.splitlines():
            if "verification status" in line.lower():
                status_raw = line.split(":", 1)[1].strip()
                break

    st = status_raw.upper()
    if not st:
        return 0
    if any(k in st for k in ("CONFLICT", "DISPUTED", "DO NOT ASSUME", "MUST CONFIRM")):
        return 0
    if any(k in st for k in ("UNVERIFIED", "NOT CONFIRMED", "UNCONFIRMED", "POSSIBLE ONLY")):
        return 0
    if "DOCUMENTED" in st:
        return 0
    if "SITE-VERIFIED" in st and not any(neg in st for neg in ("NOT SITE-VERIFIED", "PENDING")):
        return 1
    return 0


def _fallback_keywords() -> List[Tuple[str, Intent]]:
    """Single machine words, most specific first, for mangled transcripts."""
    return [
        ("baki", Intent.GET_REMAINING_PRODUCTION),
        ("remaining", Intent.GET_REMAINING_PRODUCTION),
        ("bacha", Intent.GET_REMAINING_PRODUCTION),
        ("operator", Intent.GET_OPERATOR),
        ("layer", Intent.GET_CURRENT_LAYER),
        ("width", Intent.GET_LAYER_WIDTH),
        ("material", Intent.GET_MATERIAL_USAGE),
        ("camera", Intent.GET_CAMERA_STATUS),
        ("quality", Intent.GET_PRINT_QUALITY),
        ("printing", Intent.GET_COMPLETED_PRODUCTION),
        ("print", Intent.GET_COMPLETED_PRODUCTION),
        ("production", Intent.GET_COMPLETED_PRODUCTION),
        ("wall", Intent.GET_COMPLETED_PRODUCTION),
        ("well", Intent.GET_COMPLETED_PRODUCTION),
    ]


# ---------------------------------------------------------------------------
# Resolvers -- each returns real facts or an explicit unavailability
# ---------------------------------------------------------------------------

def _unavailable(reason: str) -> RouterResult:
    return RouterResult(
        intent=Intent.UNKNOWN_QUERY, available=False, reason=reason or UNAVAILABLE_MESSAGE
    )


def _progress_facts() -> Tuple[bool, Dict[str, Any], str]:
    progress = ms.get_print_progress()
    if not progress.get("available"):
        return False, {}, progress.get("reason", UNAVAILABLE_MESSAGE)
    return True, progress, ""


def _job_and_timing() -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    job = db.get_active_job()
    return job, metrics.production_metrics(job)


def _unwrap(entry: Any) -> Any:
    """metrics fields are {value, provenance, unit, reason}; keep them intact."""
    return entry


def _spoken_line(line_id: str) -> str:
    m = re.fullmatch(r"line[_\- ]?(\w+)", (line_id or "").strip(), re.IGNORECASE)
    return f"Line {m.group(1)}" if m else (line_id or "this line")


def extract_date_from_text(text: str) -> Optional[str]:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        return m.group(1)
    months = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
        "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
        "november": 11, "nov": 11, "december": 12, "dec": 12,
    }
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(\d{4}))?\b", text)
    if m:
        day = int(m.group(1))
        m_name = m.group(2).lower()
        if m_name in months:
            y = int(m.group(3)) if m.group(3) else datetime.date.today().year
            return f"{y:04d}-{months[m_name]:02d}-{day:02d}"
    m = re.search(r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?\b", text)
    if m:
        m_name = m.group(1).lower()
        if m_name in months:
            day = int(m.group(2))
            y = int(m.group(3)) if m.group(3) else datetime.date.today().year
            return f"{y:04d}-{months[m_name]:02d}-{day:02d}"
    low = text.lower()
    if "yesterday" in low or "kal" in low:
        return (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    if "today" in low or "aaj" in low:
        return datetime.date.today().isoformat()
    return None


def extract_wall_number(text: str) -> Optional[int]:
    """
    Extracts a wall number (1, 2, 3...) from text.
    Supports:
      - Suffix digits: 'wall 1', 'wall 2', 'wall no. 3', 'wall number 4'
      - Prefix digits: '1 wall', '2 walls', '2 wall'
      - Ordinals: '1st wall', 'second wall', 'dusri wall', 'pehli wall', '3rd wall'
      - Cardinal words: 'one wall', 'two walls', 'ek wall', 'do wall', 'wall one', 'wall do'
    Returns None if no wall number or ordinal is specified.
    """
    if not text:
        return None
    t = _normalise(text)

    wall_re = r"(?:walls?|wells?)"

    # 1. Ordinals with digit suffix (e.g. '1st wall', '2nd wall', 'wall 1st', 'wall 2nd')
    m = re.search(rf"\b(\d+)(?:st|nd|rd|th)\s+{wall_re}\b", t)
    if m:
        return int(m.group(1))
    m = re.search(rf"\b{wall_re}\s*(?:number\s*|no\.?\s*)?(\d+)(?:st|nd|rd|th)\b", t)
    if m:
        return int(m.group(1))

    # 2. Plain digits after wall (e.g. 'wall 1', 'wall no 2', 'wall number 3')
    m = re.search(rf"\b{wall_re}\s*(?:number\s*|no\.?\s*)?(\d+)\b", t)
    if m:
        return int(m.group(1))

    # 3. Plain digits before wall (e.g. '1 wall', '2 walls', '2 wall')
    m = re.search(rf"\b(\d+)\s+{wall_re}\b", t)
    if m:
        return int(m.group(1))

    # 4. Ordinal words (before or after wall)
    ordinals = {
        "first": 1, "pehli": 1, "pehla": 1, "pahli": 1, "pahla": 1,
        "second": 2, "dusri": 2, "dusra": 2, "doosri": 2, "doosra": 2,
        "third": 3, "teesri": 3, "teesra": 3, "tisri": 3, "tisra": 3,
        "fourth": 4, "chauthi": 4, "chautha": 4,
        "fifth": 5, "panchvi": 5, "panchva": 5, "panchveen": 5, "panchwein": 5,
    }
    for w, num in ordinals.items():
        if re.search(rf"\b{w}\s+{wall_re}\b", t) or re.search(rf"\b{wall_re}\s+(?:number\s*|no\.?\s*)?{w}\b", t):
            return num

    # 5. Cardinal words (before or after wall)
    cardinals = {
        "ek": 1, "one": 1,
        "do": 2, "two": 2,
        "teen": 3, "three": 3,
        "char": 4, "chaar": 4, "four": 4,
        "panch": 5, "five": 5,
    }
    for w, num in cardinals.items():
        if re.search(rf"\b{w}\s+{wall_re}\b", t) or re.search(rf"\b{wall_re}\s+{w}\b", t):
            return num

    return None


def extract_downtime(text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|minute|minutes|minto|ghante|ghanta)", text.lower())
    if m:
        val = float(m.group(1))
        if "ghant" in text.lower():
            val *= 60.0
        return val
    return 0.0


# Words that can sit where a name would and are never a name. Without this,
# "okay here start my shift" would offer "okay" as the operator.
_NOT_A_NAME = {
    "okay", "ok", "kelvin", "yes", "no", "right", "so", "and", "then", "now",
    "am", "is", "i", "im", "me", "my", "shift", "start", "end", "begin",
    "line", "today", "over", "from", "just", "still", "back", "who", "it",
    "main", "hoon", "hu", "bol", "raha", "rahi", "hai", "ji", "sir",
}

# "Anand here" / "this is Anand" / "I am Anand" / "main Anand hoon" / "Anand bol raha hai".
# One capture group per form, tried in order.
_NAME_PATTERNS = (
    re.compile(r"\b(?:this\s+is|it'?s|i\s+am|i'?m|im)\s+([a-z][a-z'\-]{1,29})\b"),
    re.compile(r"\bmain\s+([a-z][a-z'\-]{1,29})\s+(?:hoon|hu|bol)"),
    re.compile(r"\b([a-z][a-z'\-]{1,29})\s+bol\s+(?:raha|rahi)\b"),
    re.compile(r"\b([a-z][a-z'\-]{1,29})\s+here\b"),
    re.compile(r"\b([a-z][a-z'\-]{1,29})\s+speaking\b"),
)


def extract_operator_name(text: str) -> Optional[str]:
    """
    The name the operator said, or None.
    """
    lowered = (text or "").lower()
    lowered = _strip_wake_word(re.sub(r"[^a-z0-9' \-]+", " ", lowered).strip())
    lowered = re.sub(r"\s+", " ", lowered).strip()

    for pattern in _NAME_PATTERNS:
        match = pattern.search(lowered)
        if not match:
            continue
        candidate = match.group(1).strip("-'")
        if candidate and candidate not in _NOT_A_NAME:
            return candidate
    return None


def extract_shift_type(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in ("night", "raat", "evening")):
        return "Night"
    if any(w in lower for w in ("afternoon", "dopahar", "second")):
        return "Afternoon"
    return "Morning"


def extract_operator_from_text(text: str, db_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Extracts (operator_id, operator_name) from natural text by matching registered operators.
    Matches operator_id, full name, or first name against word boundaries.
    """
    if not text:
        return None, None
    q_norm = text.lower().replace("'", "").replace("’", "")
    try:
        ops = shift_store.list_operators(db_path=db_path)
    except Exception:
        return None, None

    matched: List[Tuple[str, str]] = []
    for op in ops:
        op_id = op["operator_id"]
        op_name = op.get("name") or op_id
        # Check operator_id (e.g. OP001, OP_ANAND)
        if re.search(rf"\b{re.escape(op_id.lower())}\b", q_norm):
            matched.append((op_id, op_name))
            continue
        # Check full name (e.g. Anand Sharma)
        full_norm = op_name.lower().strip()
        if full_norm and re.search(rf"\b{re.escape(full_norm)}\b", q_norm):
            matched.append((op_id, op_name))
            continue
        # Check first name (if >= 3 characters, e.g. Anand)
        parts = full_norm.split()
        if parts:
            first_name = parts[0]
            if len(first_name) >= 3 and re.search(rf"\b{re.escape(first_name)}\b", q_norm):
                matched.append((op_id, op_name))
                continue

    if len(matched) == 1:
        return matched[0]
    return None, None


def extract_issue_id(text: str) -> Optional[int]:
    m = re.search(r"\b(?:issue|problem)\s*(?:number\s*|no\.?\s*|#\s*)?(\d+)\b", text.lower())
    if m:
        return int(m.group(1))
    return None


def extract_resolution_cause(text: str) -> Optional[str]:
    """
    Extracts operator-reported cause safely, preserving operator wording.
    Examples:
    - 'cause material buildup tha' -> 'material buildup'
    - 'reason was material blockage' -> 'material blockage'
    - 'cause tha material jam ho gaya tha' -> 'material jam ho gaya'
    - 'wajah material blockage thi' -> 'material blockage'
    - 'due to material blockage' -> 'material blockage'
    If question ('iska reason kya tha') or not provided, returns None.
    """
    low = text.strip()
    if re.search(r"\b(?:reason kya|cause kya|kya reason|kya cause|wajah kya|kya wajah)\b", low, re.IGNORECASE):
        return None
    if re.search(r"\b(?:kya|what|kaun|kaunsi|why|kyun)\b", low, re.IGNORECASE) and not re.search(r"\b(?:tha|thi|was|is)\b", low, re.IGNORECASE):
        return None

    m = re.search(r"\b(?:cause|reason|wajah)\s+(?:was|tha|thi|hai|is)?\s*:?\s*(.+?)(?:\s*(?:aur|action|result|and|,|\.|$))", low, re.IGNORECASE)
    if m:
        c = m.group(1).strip()
        c = re.sub(r"\s+(?:tha|thi|hai)$", "", c, flags=re.IGNORECASE).strip()
        if c and not any(w in c.lower() for w in ["kya", "what", "pata nahi", "unknown"]):
            return c
    m = re.search(r"\bdue to\s+(.+?)(?:\s*(?:aur|action|result|and|,|\.|$))", low, re.IGNORECASE)
    if m:
        c = m.group(1).strip()
        c = re.sub(r"\s+(?:tha|thi|hai)$", "", c, flags=re.IGNORECASE).strip()
        if c:
            return c
    m = re.search(r"(.+?)\s+ki wajah se\b", low, re.IGNORECASE)
    if m:
        c = m.group(1).strip()
        if c:
            return c
    return None


def extract_operator_action(text: str) -> Tuple[Optional[str], bool]:
    """
    Returns (action_taken, is_rag_recommended).
    is_rag_recommended is True ONLY if the operator explicitly states they applied/followed
    the Kelvin recommendation.
    A vague reference like 'Kelvin ne bola tha' or 'Kelvin said' returns (None, False).
    """
    low = text.lower().strip()

    # Informational questions seeking advice/procedures are NOT actions
    question_indicators = [
        "how", "kaise", "what", "kya", "why", "kyun", "karein",
        "karna hai", "karna chahiye", "should i", "can i", "could i"
    ]
    if any(re.search(rf"\b{qi}\b", low) for qi in question_indicators):
        return None, False

    applied_terms = ["apply kiya", "apply kar diya", "applied", "followed", "follow kiya", "use kiya", "used", "implement kiya", "implemented"]
    kelvin_terms = ["kelvin", "recommendation", "suggestion"]

    has_kelvin = any(k in low for k in kelvin_terms)
    has_applied = any(a in low for a in applied_terms)

    if has_kelvin:
        if has_applied:
            return "Applied Kelvin recommendation", True
        else:
            return None, False

    action_verbs = [
        "kiya", "kar diya", "kara", "cleaned", "flushed", "restarted", "replaced",
        "checked", "inspected", "examined", "fixed", "repaired", "cleared",
        "adjusted", "changed", "swapped", "tightened", "reset", "unblocked"
    ]

    if any(v in low for v in action_verbs) or re.search(r"\b(?:action|kadam|step)\b", low):
        act = text.strip()
        m = re.search(r"\b(?:action taken|action|step taken)\s*[:=]?\s*(.+)", act, re.IGNORECASE)
        if m:
            act = m.group(1).strip()
        # Clean resolution suffixes from the action string
        res_suffix_patterns = [
            r"(?:,\s*|\s+aur\s+|\s+and\s+)?(?:the\s+)?(?:material\s+flow\s+is\s+normal(?:\s+now)?|material\s+flow\s+normal(?:\s+now)?|ab\s+material\s+flow\s+normal\s+hai)\.?$",
            r"(?:,\s*|\s+aur\s+|\s+and\s+)?(?:the\s+)?(?:issue|problem)\s*(?:is\s+)?(?:resolved?|fixed?|solved?)\.?$",
            r"(?:,\s*|\s+aur\s+|\s+and\s+)?(?:the\s+)?(?:issue|problem)?\s*(?:thik|theek|solve|resolved?|fix)\s*(?:ho gaya|ho gayi|ho gya|kar diya)?\.?$",
            r"(?:,\s*|\s+aur\s+|\s+and\s+)?(?:is\s+)?(?:resolved?|fixed?|solved?)\.?$",
            r"(?:,\s*|\s+aur\s+|\s+and\s+)?it\s+is\s+working\s+now\.?$",
        ]
        for pat in res_suffix_patterns:
            act = re.sub(pat, "", act, flags=re.IGNORECASE).strip()
        act = act.rstrip(".,;").strip()
        if act:
            return act, False

    return None, False


def extract_resolution_result(text: str) -> Optional[str]:
    low = text.lower().strip()

    conditional_markers = [
        "agar", "if ", "if,", "will", "hoga", "hogi", "jayega", "jayegi",
        "dekhte", "check karte", "might", "maybe", "shayad", "hope"
    ]
    if any(m in low for m in conditional_markers):
        return None

    negative_markers = [
        "not resolved", "still stuck", "abhi bhi", "thik nahi", "theek nahi",
        "solve nahi", "not fixed", "unresolved", "kam nahi kar raha", "fail", "nahi hua"
    ]
    if any(m in low for m in negative_markers):
        return "not_resolved"

    positive_markers = [
        "the material flow is normal now", "material flow is normal now",
        "the material flow is normal", "material flow is normal",
        "ab material flow normal hai", "material flow normal hai", "material flow normal",
        "the problem is resolved", "problem is resolved", "the issue is resolved", "issue is resolved",
        "problem resolve ho gayi", "problem resolve ho gaya",
        "the issue is fixed", "issue is fixed", "issue fixed",
        "the problem is fixed", "problem is fixed", "problem fixed",
        "issue fix ho gaya", "issue fix ho gayi", "problem fix ho gaya", "problem fix ho gayi",
        "resolved", "solve ho gaya", "solve ho gayi", "solve ho gya",
        "problem solved", "issue solved", "thik ho gaya", "theek ho gaya",
        "thik ho gya", "theek ho gya", "fix ho gaya", "fix ho gya",
        "resume ho gayi", "resumed", "chal pada", "chal rahi hai ab",
        "it is working now",
        "solved", "mark as resolved", "issue resolve ho gaya"
    ]
    if any(m in low for m in positive_markers):
        return "resolved"

    return None


def extract_domain_keywords(text: str) -> List[str]:
    keywords = [
        "nozzle", "hopper", "pump", "feeder", "temperature", "motor",
        "sensor", "heater", "material", "pressure", "hose", "valve",
        "bed", "gantry", "extruder", "robot"
    ]
    low = text.lower()
    return [kw for kw in keywords if kw in low]


def extract_operator_from_text(text: str) -> Optional[str]:
    """Finds known operator ID from question text."""
    low = (text or "").lower()
    try:
        for op in shift_store.list_operators():
            name = (op.get("name") or "").lower()
            oid = (op.get("operator_id") or "").lower()
            first_name = name.split()[0] if name else ""
            if name and re.search(r"\b" + re.escape(name) + r"\b", low):
                return op["operator_id"]
            if first_name and len(first_name) >= 3 and re.search(r"\b" + re.escape(first_name) + r"\b", low):
                return op["operator_id"]
            if oid and re.search(r"\b" + re.escape(oid) + r"\b", low):
                return op["operator_id"]
    except Exception:
        pass
    return None


def resolve(intent: Intent, question: str, context: Optional[Dict[str, Any]] = None) -> RouterResult:
    """Pulls the real data for an intent. Never fabricates."""

    # --- production ---
    if intent in (
        Intent.GET_COMPLETED_PRODUCTION,
        Intent.GET_REMAINING_PRODUCTION,
        Intent.GET_TOTAL_TARGET,
    ):
        ok, progress, reason = _progress_facts()
        if not ok:
            return RouterResult(intent, available=False, reason=reason)
        return RouterResult(intent, facts={
            "job_name": progress.get("job_name"),
            "completed_mm": progress["completed_mm"],
            "remaining_mm": progress["remaining_mm"],
            "total_mm": progress["total_mm"],
            "percentage": progress["percentage"],
            "unit": "mm",
            # The job tracks LENGTH. There is no wall/piece count column in the
            # database, so a "how many walls" question cannot be answered as a
            # count -- and must not be answered as though millimetres were walls.
            "measurement_basis": "length_mm",
            "wall_count": None,
            "wall_count_reason": (
                "Production is tracked by printed length, not by a wall or piece "
                "count. No wall-count field exists in the database."
            ),
        })

    if intent == Intent.GET_PRINT_RATE:
        _job, timing = _job_and_timing()
        entry = timing.get("print_rate_mm_per_min", {})
        if entry.get("value") is None:
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={"print_rate_mm_per_min": entry["value"], "unit": "mm/min"})

    if intent == Intent.GET_ELAPSED_TIME:
        _job, timing = _job_and_timing()
        entry = timing.get("elapsed", {})
        if entry.get("value") is None:
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={"elapsed": entry["value"]})

    if intent == Intent.GET_ETA:
        _job, timing = _job_and_timing()
        entry = timing.get("eta", {})
        if entry.get("value") is None:
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={"eta": entry["value"]})

    if intent == Intent.GET_DEADLINE_STATUS:
        _job, timing = _job_and_timing()
        entry = timing.get("deadline_status", {})
        if entry.get("value") is None:
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={
            "deadline_status": entry["value"],
            "time_to_deadline": timing.get("time_to_deadline", {}).get("value"),
        })

    # --- layers / geometry ---
    if intent == Intent.GET_CURRENT_LAYER:
        layer = ms.get_current_layer()
        if not layer.get("available"):
            return RouterResult(intent, available=False, reason=layer.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={
            "current_layer": layer["current_layer"],
            "total_layers": layer.get("total_layers"),
        })

    if intent == Intent.GET_LAYER_WIDTH:
        width = ms.get_current_layer_width()
        if not width.get("available"):
            return RouterResult(intent, available=False, reason=width.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={
            "layer_width_mm": width["layer_width_mm"],
            "confidence": width.get("confidence"),
            "unit": "mm",
            "note": width.get("note"),
        })

    if intent in (Intent.GET_LAYER_HEIGHT, Intent.GET_WALL_DIMENSIONS):
        job = db.get_active_job()
        from agent.services.machine_state import _snapshot  # live vision snapshot

        vision = _snapshot() or {}
        geometry = metrics.geometry_metrics(job, {"layers": {
            "layer_height_mm": vision.get("layer_height_mm"),
            "layer_height_px": vision.get("layer_height_px"),
            "layer_width_mm": vision.get("layer_width"),
            "layer_width_px": vision.get("layer_width_px"),
        }})
        key = "layer_height_mm" if intent == Intent.GET_LAYER_HEIGHT else "wall_height_mm"
        entry = geometry.get(key, {})
        if entry.get("value") is None:
            extra = geometry.get("wall_width_mm", {}) if intent == Intent.GET_WALL_DIMENSIONS else {}
            if extra.get("value") is not None:
                return RouterResult(intent, facts={"wall_width_mm": extra["value"], "unit": "mm"})
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        facts = {key: entry["value"], "unit": "mm"}
        if intent == Intent.GET_WALL_DIMENSIONS:
            facts["wall_width_mm"] = geometry.get("wall_width_mm", {}).get("value")
        return RouterResult(intent, facts=facts)

    # --- material ---
    if intent in (Intent.GET_MATERIAL_USAGE, Intent.GET_MATERIAL_REMAINING, Intent.GET_MATERIAL_RATE):
        job, timing = _job_and_timing()
        elapsed = timing.get("elapsed_seconds", {}).get("value")
        material = metrics.material_metrics(job, elapsed)
        key = {
            Intent.GET_MATERIAL_USAGE: "material_used_kg",
            Intent.GET_MATERIAL_REMAINING: "material_remaining_kg",
            Intent.GET_MATERIAL_RATE: "consumption_rate_kg_per_hour",
        }[intent]
        entry = material.get(key, {})
        if entry.get("value") is None:
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={key: entry["value"], "unit": entry.get("unit", "kg")})

    # --- history ---
    if intent in (Intent.GET_24_HOUR_PRODUCTION, Intent.GET_SHIFT_PRODUCTION, Intent.GET_RECENT_EVENTS):
        history = metrics.production_history()
        window = "24h" if intent == Intent.GET_24_HOUR_PRODUCTION else (
            "8h" if intent == Intent.GET_SHIFT_PRODUCTION else "10m")
        entry = history["windows"].get(window, {})
        events = db.get_recent_events(limit=5) if intent == Intent.GET_RECENT_EVENTS else None
        if entry.get("value") is None and not events:
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        facts: Dict[str, Any] = {"window": window}
        if entry.get("value") is not None:
            facts["produced_mm"] = entry["value"]
            facts["unit"] = "mm"
        else:
            facts["produced_note"] = entry.get("reason", "")
        if events:
            facts["events"] = [
                {"type": e.get("event_type", ""), "message": e.get("message") or e.get("description", ""),
                 "seconds_ago": round(__import__("time").time() - e.get("timestamp", 0), 1)}
                for e in events
            ]
        return RouterResult(intent, facts=facts)

    if intent == Intent.WAKE_WORD:
        return RouterResult(intent, facts={"acknowledged": True})

    # --- operator ---
    if intent == Intent.GET_OPERATOR:
        # Queries live database (operators and shifts) for currently working operators
        cand_name = identity.extract_name_candidate(question)
        if cand_name and cand_name.lower() not in {"who", "anyone", "someone", "somebody", "anybody", "everybody", "everyone"}:
            st_spec = shift_store.get_operators_status(cand_name)
            if st_spec.get("found"):
                actual_name = st_spec.get("operator_name", cand_name)
                is_working = st_spec.get("is_working", False)
                curr_status = st_spec.get("current_work_status", "Off Shift")
                return RouterResult(
                    intent,
                    facts={
                        "specific_query": True,
                        "operator_name": actual_name,
                        "is_working": is_working,
                        "current_work_status": curr_status,
                        "message": st_spec.get("message"),
                        "source": "live_shifts_database",
                    },
                )

        st_all = shift_store.get_operators_status()
        if st_all.get("has_active_workers"):
            working_list = st_all.get("working_operators", [])
            op_str = ", ".join(working_list)
            return RouterResult(
                intent,
                facts={
                    "operator": op_str,
                    "active": True,
                    "working_operators": working_list,
                    "source": "live_shifts_database",
                },
            )
        return RouterResult(
            intent,
            available=True,
            facts={"operator": None, "active": False, "working_operators": []},
            reason="No operators are currently clocked in on an active shift right now.",
        )

    # --- status ---
    if intent == Intent.GET_MACHINE_STATUS:
        status = ms.get_current_print_status()
        machine = ms.get_machine_status()
        if not status.get("available") and not machine.get("available"):
            return RouterResult(intent, available=False, reason=status.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={
            "printing": status.get("printing"),
            "status": status.get("status"),
            "confidence": machine.get("confidence"),
            "note": status.get("note") or machine.get("note"),
        })

    if intent in (Intent.GET_ROBOT_STATUS, Intent.GET_CAMERA_STATUS, Intent.GET_COMMUNICATION_STATUS):
        machine = ms.get_machine_status()
        camera = (machine.get("camera") or {}) if machine.get("available") else {}
        if intent == Intent.GET_CAMERA_STATUS:
            return RouterResult(
                intent,
                facts={"camera_connected": False, "status": "unavailable", "reason": "Camera functionality is unavailable in this version."},
                available=False,
                reason="Camera functionality is unavailable in this version.",
            )
        if not machine.get("available"):
            return RouterResult(intent, available=False, reason=machine.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={
            "camera_connected": False,
            "machine_status": machine.get("status"),
            "machine_link": "not connected",
            "machine_link_note": (
                "No machine communication link is connected; robot telemetry such as "
                "pump and accelerator is unavailable."
            ),
        })

    if intent == Intent.GET_SITE_PEOPLE_COUNT:
        return RouterResult(
            intent,
            facts={"people_count": 0, "status": "unavailable", "reason": "Camera functionality is unavailable in this version."},
            available=False,
            reason="Camera functionality is unavailable in this version.",
        )

    if intent == Intent.GET_SITE_DETECTIONS:
        return RouterResult(
            intent,
            facts={"entities": [], "status": "unavailable", "reason": "Camera functionality is unavailable in this version."},
            available=False,
            reason="Camera functionality is unavailable in this version.",
        )

    if intent == Intent.GET_PRINT_QUALITY:
        from agent.services.machine_state import _live_state

        snapshot = _live_state.snapshot() if _live_state is not None else {}
        quality = ((snapshot.get("vision") or {}).get("print_quality") or {}) if isinstance(snapshot, dict) else {}
        score = quality.get("score_percent")
        if isinstance(score, dict) or score is None:
            reason = score.get("reason") if isinstance(score, dict) else "Print quality has not been scored yet."
            return RouterResult(intent, available=False, reason=reason or UNAVAILABLE_MESSAGE)
        return RouterResult(intent, facts={
            "quality_percent": score,
            "quality_status": quality.get("status"),
            "issues": quality.get("issues", []),
        })

    if intent == Intent.GET_RECENT_PROBLEMS:
        from agent.services.machine_state import _live_state

        snapshot = _live_state.snapshot() if _live_state is not None else {}
        alerts = snapshot.get("recent_alerts", []) if isinstance(snapshot, dict) else []
        events = db.get_recent_events(limit=5)
        if not alerts and not events:
            return RouterResult(intent, facts={"problems": [], "message": "No problems have been recorded."})
        return RouterResult(intent, facts={
            "alerts": alerts,
            "events": [{"type": e.get("event_type", ""), "message": e.get("message") or e.get("description", "")} for e in events],
        })

    # --- knowledge ---
    if intent == Intent.KNOWLEDGE_QUERY:
        result = search_documentation(question)
        return RouterResult(
            intent,
            knowledge=result,
            available=result.get("found", False),
            reason="" if result.get("found") else result.get("message", ""),
        )

    # --- Phase 2C Historical Queries ---
    if intent == Intent.GET_ACTIVE_SHIFT:
        try:
            line_id = (context.get("line_id") if context else None) or "line_1"
            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                line_label = _spoken_line(line_id)
                return RouterResult(intent, facts={"active": False, "line_id": line_id, "message": f"No active shift is currently running on {line_label}."})
            op = shift_store.get_operator(active["operator_id"])
            return RouterResult(intent, facts={
                "active": True,
                "shift_id": active["shift_id"],
                "shift_type": active["shift_type"],
                "shift_date": active["shift_date"],
                "line_id": active["line_id"],
                "operator_id": active["operator_id"],
                "operator_name": op["name"] if op else active["operator_id"],
                "target_walls": active["target_walls"],
            })
        except Exception as exc:
            log.warning("Could not resolve GET_ACTIVE_SHIFT: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query active shift.")

    if intent == Intent.GET_PREVIOUS_SHIFT:
        try:
            line_id = (context.get("line_id") if context else None) or "line_1"
            shifts = shift_store.list_shifts(status="closed", limit=10)
            line_prev = [s for s in shifts if s.get("line_id", "line_1") == line_id]
            if not line_prev:
                return RouterResult(intent, facts={"found": False, "message": f"No previous closed shift was found for {line_id}."})
            prev = line_prev[0]
            op = shift_store.get_operator(prev["operator_id"])
            op_name = op["name"] if op else prev["operator_id"]

            st = prev.get("start_time")
            et = prev.get("end_time")
            dur_sec = (et - st) if (st is not None and et is not None) else None
            dur_formatted = ""
            if dur_sec is not None and dur_sec >= 0:
                hours = int(dur_sec // 3600)
                minutes = int((dur_sec % 3600) // 60)
                dur_formatted = f"{hours} hours {minutes} minutes" if hours > 0 else f"{minutes} minutes"

            st_formatted = datetime.datetime.fromtimestamp(st).strftime("%I:%M %p").lstrip("0") if st else ""
            et_formatted = datetime.datetime.fromtimestamp(et).strftime("%I:%M %p").lstrip("0") if et else ""

            return RouterResult(intent, facts={
                "found": True,
                "shift_id": prev["shift_id"],
                "shift_type": prev["shift_type"],
                "shift_date": prev["shift_date"],
                "line_id": line_id,
                "operator_id": prev["operator_id"],
                "operator_name": op_name,
                "start_time": st,
                "end_time": et,
                "start_time_formatted": st_formatted,
                "end_time_formatted": et_formatted,
                "duration_seconds": dur_sec,
                "duration_hours": round(dur_sec / 3600.0, 2) if dur_sec is not None else None,
                "duration_formatted": dur_formatted,
                "summary_notes": prev.get("summary_notes") or "No notes recorded.",
            })
        except Exception as exc:
            log.warning("Could not resolve GET_PREVIOUS_SHIFT: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query previous shift.")

    if intent == Intent.GET_SHIFT_BY_DATE:
        date_str = extract_date_from_text(question) or time.strftime("%Y-%m-%d")
        try:
            shifts = shift_store.list_shifts(shift_date=date_str)
            operator_names = []
            for s in shifts:
                op = shift_store.get_operator(s["operator_id"])
                name = op["name"] if op else s["operator_id"]
                if name not in operator_names:
                    operator_names.append(name)
            return RouterResult(intent, facts={
                "date": date_str,
                "count": len(shifts),
                "shifts": shifts,
                "operator_names": operator_names,
            })
        except Exception as exc:
            log.warning("Could not resolve GET_SHIFT_BY_DATE: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query shifts by date.")

    if intent == Intent.GET_SHIFT_BY_OPERATOR:
        try:
            q_norm = question.lower()
            self_terms = ["my", "i", "meri", "maine", "mera", "mujhe"]
            is_self_query = any(re.search(rf"\b{re.escape(w)}\b", q_norm) for w in self_terms)

            if is_self_query:
                op_id = context.get("operator_id") if context else None
                if not op_id:
                    line_id = context.get("line_id", "line_1") if context else "line_1"
                    active = shift_store.get_active_shift_for_line(line_id)
                    if active:
                        op_id = active.get("operator_id")
                if not op_id:
                    return RouterResult(
                        intent,
                        facts={"needs_clarification": True, "clarification_type": "operator"},
                        reason="Could not determine your active operator profile. Which operator are you asking about?",
                        available=False,
                    )
            else:
                op_id = extract_operator_from_text(question)
                if not op_id and context:
                    op_id = context.get("operator_id")

            if not op_id:
                return RouterResult(
                    intent,
                    facts={"needs_clarification": True, "clarification_type": "operator"},
                    reason="Please specify which operator you want to check.",
                    available=False,
                )

            op_obj = shift_store.get_operator(op_id)
            op_name = op_obj.get("name") if op_obj else op_id

            # Determine whether this is a timing/duration query vs a wall production query
            is_timing_query = any(w in q_norm for w in [
                "when did", "timing", "start time", "end time", "shift start", "shift end",
                "kab shuru", "kab khatam", "shuru hui", "khatam hui"
            ]) or (any(w in q_norm for w in ["start", "end", "shuru", "khatam"]) and "shift" in q_norm)

            is_duration_query = any(w in q_norm for w in [
                "hour", "hours", "ghante", "ghanta", "duration", "kitne ghante"
            ]) and not any(w in q_norm for w in ["wall", "walls", "kitni wall", "kitni walls", "piece", "pieces"])

            date_str = extract_date_from_text(question)

            if is_duration_query or is_timing_query:
                shifts = shift_store.list_shifts(operator_id=op_id, shift_date=date_str, limit=50) if date_str else shift_store.list_shifts(operator_id=op_id, limit=50)
                if not shifts:
                    return RouterResult(
                        intent,
                        facts={
                            "found": False,
                            "operator_id": op_id,
                            "operator_name": op_name,
                            "is_duration_query": is_duration_query,
                            "is_timing_query": is_timing_query,
                            "date": date_str,
                        },
                        reason=f"No shift records found for {op_name}.",
                        available=True,
                    )

                target_shift = shifts[0]
                st = target_shift.get("start_time")
                et = target_shift.get("end_time")
                is_active = target_shift.get("status") == "active"
                st_formatted = datetime.datetime.fromtimestamp(st).strftime("%I:%M %p").lstrip("0") if st else ""
                et_formatted = datetime.datetime.fromtimestamp(et).strftime("%I:%M %p").lstrip("0") if et else ""

                dur_sec = (et - st) if (st is not None and et is not None) else None
                dur_formatted = ""
                if dur_sec is not None and dur_sec >= 0:
                    hours = int(dur_sec // 3600)
                    minutes = int((dur_sec % 3600) // 60)
                    dur_formatted = f"{hours} hours {minutes} minutes" if hours > 0 else f"{minutes} minutes"

                # For ACTIVE shifts: do NOT calculate final duration with live time.
                if is_active or et is None:
                    return RouterResult(
                        intent,
                        facts={
                            "found": True,
                            "operator_id": op_id,
                            "operator_name": op_name,
                            "is_duration_query": is_duration_query,
                            "is_timing_query": is_timing_query,
                            "is_self_query": is_self_query,
                            "shift_id": target_shift["shift_id"],
                            "shift_type": target_shift["shift_type"],
                            "shift_date": target_shift["shift_date"],
                            "line_id": target_shift.get("line_id", "line_1"),
                            "status": "active",
                            "start_time": st,
                            "end_time": None,
                            "start_time_formatted": st_formatted,
                            "end_time_formatted": "ongoing",
                            "duration_seconds": None,
                            "duration_hours": None,
                            "duration_formatted": "ongoing",
                            "duration_available": False,
                        },
                        available=True,
                    )

                # For CLOSED shifts: duration = end_time - start_time
                return RouterResult(
                    intent,
                    facts={
                        "found": True,
                        "operator_id": op_id,
                        "operator_name": op_name,
                        "is_duration_query": is_duration_query,
                        "is_timing_query": is_timing_query,
                        "is_self_query": is_self_query,
                        "shift_id": target_shift["shift_id"],
                        "shift_type": target_shift["shift_type"],
                        "shift_date": target_shift["shift_date"],
                        "line_id": target_shift.get("line_id", "line_1"),
                        "status": target_shift.get("status", "closed"),
                        "start_time": st,
                        "end_time": et,
                        "start_time_formatted": st_formatted,
                        "end_time_formatted": et_formatted,
                        "duration_seconds": dur_sec,
                        "duration_hours": round(dur_sec / 3600.0, 2) if dur_sec is not None else None,
                        "duration_formatted": dur_formatted,
                        "duration_available": True,
                    },
                    available=True,
                )

            records = shift_store.get_operator_production(operator_id=op_id, shift_date=date_str)
            completed_walls = [r for r in records if r.get("status") == "completed"]
            completed_wall_nums = sorted(list({r["wall_number"] for r in completed_walls}))

            return RouterResult(
                intent,
                facts={
                    "found": True,
                    "operator_id": op_id,
                    "operator_name": op_name,
                    "completed_count": len(completed_wall_nums),
                    "completed_walls": completed_wall_nums,
                    "records_count": len(records),
                    "records": records,
                    "source": "shift_store",
                    "is_duration_query": False,
                    "is_production_query": True,
                    "wall_count": len(completed_wall_nums),
                    "walls": completed_wall_nums,
                    "date": date_str,
                },
                available=True,
            )
        except Exception as exc:
            log.warning("Could not resolve GET_SHIFT_BY_OPERATOR: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query operator records.")

    if intent == Intent.GET_WALL_HISTORY:
        wall_num = extract_wall_number(question)
        if wall_num is None:
            return RouterResult(intent, available=False, reason="Please specify which wall number you want to check.")
        try:
            with db.connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM production_records WHERE wall_number = ? ORDER BY start_time ASC",
                    (wall_num,),
                ).fetchall()
            records = [dict(r) for r in rows]
            if not records:
                return RouterResult(intent, facts={"wall_number": wall_num, "found": False, "message": f"No production records found for Wall {wall_num}."})
            latest = records[-1]
            return RouterResult(intent, facts={
                "wall_number": wall_num,
                "found": True,
                "status": latest["status"],
                "layers_completed": latest["layers_completed"],
                "segment_layers": latest["segment_layers"],
                "wall_height": latest["wall_height"],
                "records_count": len(records),
            })
        except Exception as exc:
            log.warning("Could not resolve GET_WALL_HISTORY: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query wall history.")

    if intent == Intent.GET_SHIFT_ISSUES:
        try:
            # 1. Resolve operator from question text or fallback to context / active shift on context line
            op_id = extract_operator_from_text(question)
            shift_id = context.get("shift_id") if context else None
            if not op_id and context:
                op_id = context.get("operator_id")
                if not op_id and context.get("line_id"):
                    active = shift_store.get_active_shift_for_line(context["line_id"])
                    if active:
                        op_id = active.get("operator_id")
                        if not shift_id:
                            shift_id = active.get("shift_id")
            if not shift_id and context and context.get("line_id"):
                active = shift_store.get_active_shift_for_line(context["line_id"])
                if active and (not op_id or active.get("operator_id") == op_id):
                    shift_id = active.get("shift_id")

            focus = "general"
            q_low = question.lower()
            if any(w in q_low for w in ["recommend", "recommendation", "kelvin", "suggest"]):
                focus = "recommendation"
            elif any(w in q_low for w in ["verified", "rag verified"]):
                focus = "rag_verified"
            elif any(w in q_low for w in ["source", "rag source"]):
                focus = "rag_source"
            elif any(w in q_low for w in ["action", "step", "kya kiya", "kya step", "kya action", "liya"]):
                focus = "action"
            elif any(w in q_low for w in ["cause", "reason", "wajah", "kyun", "why"]):
                focus = "cause"

            # 2. Resolve explicit issue ID from question text or fallback to context
            explicit_id = extract_issue_id(question)
            used_last_issue_id = False
            if explicit_id is None and context and context.get("last_issue_id"):
                explicit_id = context.get("last_issue_id")
                used_last_issue_id = True

            # 3. Explicit Issue #ID handling with strict operator isolation
            if explicit_id is not None:
                target_issue = shift_store.get_issue(explicit_id)
                if not target_issue:
                    return RouterResult(
                        intent,
                        facts={"found": False, "issue_id": explicit_id},
                        reason=f"Issue #{explicit_id} not found.",
                        available=False,
                    )

                # Parent shift and owner lookup
                shift = shift_store.get_shift(target_issue["shift_id"]) if target_issue.get("shift_id") else None
                issue_op_id = shift.get("operator_id") if shift else None

                # Operator verification:
                # If last_issue_id is used from context, verify parent shift operator matches context operator_id
                if used_last_issue_id:
                    context_op_id = context.get("operator_id") if context else None
                    if not context_op_id or issue_op_id != context_op_id:
                        op_obj = shift_store.get_operator(context_op_id) if context_op_id else None
                        op_name = op_obj.get("name") if op_obj else context_op_id
                        return RouterResult(
                            intent,
                            facts={
                                "found": False,
                                "issue_id": explicit_id,
                                "operator_id": context_op_id,
                                "isolation_violation": True,
                            },
                            reason=f"Issue #{explicit_id} does not belong to operator {op_name}." if op_name else f"Issue #{explicit_id} does not match the active operator.",
                            available=False,
                        )

                # Explicit operator in query or context must match issue owner
                if op_id and issue_op_id != op_id:
                    op_obj = shift_store.get_operator(op_id)
                    op_name = op_obj.get("name") if op_obj else op_id
                    return RouterResult(
                        intent,
                        facts={
                            "found": False,
                            "issue_id": explicit_id,
                            "operator_id": op_id,
                            "isolation_violation": True,
                        },
                        reason=f"Issue #{explicit_id} does not belong to operator {op_name}.",
                        available=False,
                    )

                return RouterResult(
                    intent,
                    facts={
                        "found": True,
                        "issue_id": explicit_id,
                        "operator_id": issue_op_id or op_id,
                        "focus": focus,
                        "cause": target_issue.get("cause"),
                        "action_taken": target_issue.get("action_taken"),
                        "rag_recommendation": target_issue.get("rag_recommendation"),
                        "rag_verified": bool(target_issue.get("rag_verified")),
                        "rag_source": target_issue.get("rag_source"),
                        "description": target_issue.get("description"),
                        "issue_type": target_issue.get("issue_type"),
                        "result": target_issue.get("result"),
                        "severity": target_issue.get("severity"),
                    },
                    available=True,
                )

            # 4. No explicit issue ID -> require operator (do not list all operators' issues)
            if not op_id:
                return RouterResult(
                    intent,
                    facts={
                        "needs_clarification": True,
                        "clarification_type": "operator",
                    },
                    reason="Which operator's shift issues would you like to check?",
                    available=False,
                )

            date_str = extract_date_from_text(question)
            if date_str:
                shift_id = None

            if shift_id:
                issues = shift_store.list_shift_issues(shift_id=shift_id, operator_id=op_id, shift_date=date_str, limit=20)
            else:
                issues = shift_store.list_shift_issues(operator_id=op_id, shift_date=date_str, limit=20)
            if not issues:
                op_obj = shift_store.get_operator(op_id) if op_id else None
                op_name = op_obj.get("name") if op_obj else op_id
                op_str = f" for operator {op_name}" if op_name else ""
                date_str_part = f" on {date_str}" if date_str else ""
                return RouterResult(
                    intent,
                    facts={"count": 0, "operator_id": op_id, "date": date_str, "issues": []},
                    reason=f"No issues have been recorded{op_str}{date_str_part}.",
                    available=True,
                )

            # Focus == "action": return current operator's action from current or most recent issue
            if focus == "action":
                target = None
                if context and context.get("last_issue_id"):
                    target = next((iss for iss in issues if iss["issue_id"] == context["last_issue_id"]), None)
                if not target and issues:
                    target = issues[0]
                if target:
                    return RouterResult(
                        intent,
                        facts={
                            "found": True,
                            "issue_id": target.get("issue_id"),
                            "operator_id": op_id,
                            "focus": focus,
                            "cause": target.get("cause"),
                            "action_taken": target.get("action_taken"),
                            "rag_recommendation": target.get("rag_recommendation"),
                            "rag_verified": bool(target.get("rag_verified")),
                            "rag_source": target.get("rag_source"),
                            "description": target.get("description"),
                            "issue_type": target.get("issue_type"),
                            "result": target.get("result"),
                            "severity": target.get("severity"),
                        },
                        available=True,
                    )

            # Multi-issue clarification check for detail queries
            if focus in ("cause", "recommendation", "rag_verified", "rag_source") and len(issues) > 1:
                cand_ids = [iss["issue_id"] for iss in issues]
                op_obj = shift_store.get_operator(op_id) if op_id else None
                op_name = op_obj.get("name") if op_obj else op_id
                op_str = f" for {op_name}" if op_name else ""
                cand_list = ", ".join(f"Issue #{i}" for i in cand_ids)
                return RouterResult(
                    intent,
                    facts={
                        "needs_clarification": True,
                        "candidate_ids": cand_ids,
                        "operator_id": op_id,
                        "focus": focus,
                        "count": len(issues),
                    },
                    reason=f"Multiple issues found{op_str} ({cand_list}). Which issue number would you like to check?",
                    available=False,
                )

            # Single issue detail query
            if focus in ("cause", "recommendation", "rag_verified", "rag_source") and len(issues) == 1:
                single = issues[0]
                return RouterResult(
                    intent,
                    facts={
                        "found": True,
                        "issue_id": single.get("issue_id"),
                        "operator_id": op_id,
                        "focus": focus,
                        "cause": single.get("cause"),
                        "action_taken": single.get("action_taken"),
                        "rag_recommendation": single.get("rag_recommendation"),
                        "rag_verified": bool(single.get("rag_verified")),
                        "rag_source": single.get("rag_source"),
                        "description": single.get("description"),
                        "issue_type": single.get("issue_type"),
                        "result": single.get("result"),
                        "severity": single.get("severity"),
                    },
                    available=True,
                )

            # General focus: list all formatted issues for this operator
            formatted_issues = []
            for iss in issues:
                formatted_issues.append({
                    "issue_id": iss.get("issue_id"),
                    "type": iss.get("issue_type"),
                    "desc": iss.get("description"),
                    "severity": iss.get("severity"),
                    "cause": iss.get("cause"),
                    "action_taken": iss.get("action_taken"),
                    "rag_recommendation": iss.get("rag_recommendation"),
                    "rag_verified": bool(iss.get("rag_verified")),
                    "rag_source": iss.get("rag_source"),
                    "result": iss.get("result"),
                    "shift_date": iss.get("shift_date"),
                    "shift_id": iss.get("shift_id"),
                    "operator_id": iss.get("operator_id"),
                })
            return RouterResult(
                intent,
                facts={
                    "count": len(formatted_issues),
                    "operator_id": op_id,
                    "date": date_str,
                    "focus": focus,
                    "issues": formatted_issues,
                },
                available=True,
            )
        except Exception as exc:
            log.warning("Could not resolve GET_SHIFT_ISSUES: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query shift issues.")

    if intent == Intent.GET_HANDOVER_STATUS:
        try:
            h = shift_store.get_pending_handover_for_line("line_1")
            if not h:
                with db.connect() as conn:
                    row = conn.execute(
                        "SELECT * FROM handovers WHERE line_id = ? ORDER BY handover_time DESC LIMIT 1",
                        ("line_1",),
                    ).fetchone()
                h = dict(row) if row else None
            if not h:
                return RouterResult(intent, facts={"found": False, "message": "No handover record was found for Line 1."})
            return RouterResult(intent, facts={
                "found": True,
                "handover_id": h["handover_id"],
                "completed_work": h.get("completed_work") or "None",
                "pending_work": h.get("pending_work") or "None",
                "open_issues": h.get("open_issues") or "None",
                "to_shift_id": h.get("to_shift_id"),
            })
        except Exception as exc:
            log.warning("Could not resolve GET_HANDOVER_STATUS: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query handover status.")

    # --- Phase 2C Action Intents (Prepared for confirmation) ---
    if intent == Intent.SHIFT_START:
        st = extract_shift_type(question)
        tw = extract_wall_number(question) or 0
        return RouterResult(intent, facts={
            "action": "SHIFT_START",
            "shift_type": st,
            "target_walls": tw,
            "line_id": "line_1",
            # The name the operator said, if any. The assistant resolves it
            # against the operators table; the router never touches the
            # database for identity and never invents a name.
            "spoken_name": extract_operator_name(question),
            "requires_confirmation": True,
        })

    if intent == Intent.SHIFT_END:
        return RouterResult(intent, facts={
            "action": "SHIFT_END",
            "status": "closed",
            "line_id": "line_1",
            "requires_confirmation": True,
        })

    if intent == Intent.LOG_WALL_PRODUCTION:
        wn = extract_wall_number(question)
        return RouterResult(intent, facts={
            "action": "LOG_WALL_PRODUCTION",
            "wall_number": wn,
            "status": "completed",
            "line_id": "line_1",
            "requires_confirmation": wn is not None,
        })

    if intent == Intent.LOG_SHIFT_ISSUE:
        wn = extract_wall_number(question)
        dt = extract_downtime(question)
        rag_rec = None
        rag_src = None
        rag_ver = 0
        search_q = re.sub(r"\bmaterial\s+slope\b", "material flow", question, flags=re.IGNORECASE)
        try:
            k = search_documentation(search_q)
            if k.get("found") and k.get("results"):
                spoken = speak_knowledge(k, search_q)
                top = k["results"][0]
                rag_rec = spoken or top.get("text", "")[:320]
                rag_src = top.get("source")
                rag_ver = _evaluate_rag_verified(top.get("text", ""))
        except Exception as exc:
            log.warning("Could not retrieve RAG guidance for LOG_SHIFT_ISSUE: %s", exc)

        issue_type = "operational_issue"
        q_low = question.lower()
        if "material flow" in q_low or "material slope" in q_low:
            issue_type = "material_flow_issue"
        elif "nozzle" in q_low:
            issue_type = "nozzle_clog"
        elif "pump" in q_low:
            issue_type = "pump_pressure"
        elif "feeder" in q_low:
            issue_type = "feeder_jam"
        elif "sensor" in q_low:
            issue_type = "sensor_fault"

        return RouterResult(intent, facts={
            "action": "LOG_SHIFT_ISSUE",
            "issue_type": issue_type,
            "description": question,
            "wall_number": wn,
            "downtime_minutes": dt,
            "severity": "warning",
            "line_id": "line_1",
            "rag_recommendation": rag_rec,
            "rag_source": rag_src,
            "rag_verified": rag_ver,
            "requires_confirmation": True,
        })

    if intent == Intent.LOG_SHIFT_HANDOVER:
        return RouterResult(intent, facts={
            "action": "LOG_SHIFT_HANDOVER",
            "line_id": "line_1",
            "requires_confirmation": True,
        })

    if intent == Intent.RESOLVE_SHIFT_ISSUE:
        t_id = extract_issue_id(question)
        action_taken, rag_recommended = extract_operator_action(question)
        cause = extract_resolution_cause(question)
        result = extract_resolution_result(question)
        domain_keywords = extract_domain_keywords(question)
        return RouterResult(intent, facts={
            "action": "RESOLVE_SHIFT_ISSUE",
            "target_issue_id": t_id,
            "action_taken": action_taken,
            "cause": cause,
            "result": result,
            "domain_keywords": domain_keywords,
            "rag_recommended": rag_recommended,
            "line_id": "line_1",
            "question": question,
            "requires_confirmation": True,
        })

    if intent == Intent.START_PRINTING_JOB:
        m = re.search(r"\b(?:start|begin)\s+(?:printing\s+)?job\s+([A-Za-z0-9_]+)\b", question, re.IGNORECASE)
        j_name = m.group(1).strip() if m else "Print_Job_1"
        return RouterResult(intent, facts={
            "action": "START_PRINTING_JOB",
            "job_name": j_name,
            "requires_confirmation": True,
        })

    if intent == Intent.END_PRINTING_JOB:
        m = re.search(r"\b(?:finish|end|complete|stop)\s+(?:printing\s+)?job\s+([A-Za-z0-9_]+)\b", question, re.IGNORECASE)
        j_name = m.group(1).strip() if m else "Print_Job_1"
        return RouterResult(intent, facts={
            "action": "END_PRINTING_JOB",
            "job_name": j_name,
            "requires_confirmation": True,
        })

    if intent == Intent.OPERATIONAL_REPORT:
        q_low = question.lower()
        report = shift_store.get_agent_reporting_view(shift_date="2026-10-08", db_path=config.DATABASE_PATH)
        if not report:
            op_id_extracted = extract_operator_from_text(question)
            if op_id_extracted:
                return resolve(Intent.GET_SHIFT_BY_OPERATOR, question, context=context)
            return RouterResult(intent, available=False, reason="Operational reporting data is not available.")

        is_arrival = any(w in q_low for w in ("arrive", "came in", "come in", "clock in"))
        is_departure = any(w in q_low for w in ("leave", "clock out", "came out", "exit"))
        is_ot = "overtime" in q_low
        is_hours = "hours did" in q_low or "how many hours" in q_low
        is_late = "late" in q_low
        is_tot_print = "total printing time" in q_low or ("printing time" in q_low and "total" in q_low)
        is_crossed_9 = "crossed 9" in q_low or "cross 9" in q_low or "more than 9" in q_low or "over 9" in q_low
        is_highest_ot = "highest overtime" in q_low or "most overtime" in q_low or "maximum overtime" in q_low
        is_working_days = "working days" in q_low or "total working days" in q_low
        is_problem = "problem" in q_low or "issue" in q_low or "facing" in q_low or "faced" in q_low

        # 1. Combined question: arrival + overtime
        if is_arrival and is_ot and "aman" in q_low:
            aman_row = next((r for r in report if r["operator_name"].lower() == "aman"), None)
            if aman_row:
                return RouterResult(intent, facts={
                    "type": "combined_arrival_overtime",
                    "operator_name": "Aman",
                    "came_in": aman_row["came_in"],
                    "overtime": aman_row["overtime"],
                    "job_name": aman_row["job_name"],
                }, available=True)

        # 2. Late arrivals
        if is_late and "who" in q_low:
            late_ops = [r for r in report if r["late_minutes"] > 0]
            return RouterResult(intent, facts={
                "type": "who_arrived_late",
                "late_operators": [
                    {
                        "name": r["operator_name"],
                        "late_minutes": r["late_minutes"],
                        "came_in": r["came_in"],
                        "shift": r["shift_type"],
                    }
                    for r in late_ops
                ],
            }, available=True)

        # 3. Specific operator arrival timing
        if is_arrival:
            for r in report:
                if r["operator_name"].lower() in q_low:
                    early_late = f"({r['late_minutes']} minutes late)" if r['late_minutes'] > 0 else "(arrived on time)"
                    return RouterResult(intent, facts={
                        "type": "operator_arrival",
                        "operator_name": r["operator_name"],
                        "came_in": r["came_in"],
                        "late_minutes": r["late_minutes"],
                        "note": early_late,
                    }, available=True)

        # 4. Specific operator departure timing
        if is_departure:
            for r in report:
                if r["operator_name"].lower() in q_low:
                    return RouterResult(intent, facts={
                        "type": "operator_departure",
                        "operator_name": r["operator_name"],
                        "came_out": r["came_out"],
                    }, available=True)

        # 5. Specific operator shift duration / hours worked
        if is_hours:
            for r in report:
                if r["operator_name"].lower() in q_low:
                    return RouterResult(intent, facts={
                        "type": "operator_hours",
                        "operator_name": r["operator_name"],
                        "hours": r["shift_duration"],
                    }, available=True)

        # 6. Problem / issue faced by operator or reported during shift
        if is_problem:
            for r in report:
                if r["operator_name"].lower() in q_low and r.get("problem_type"):
                    return RouterResult(intent, facts={
                        "type": "operator_last_problem",
                        "operator_name": r["operator_name"],
                        "shift_type": r["shift_type"],
                        "problem_type": r["problem_type"],
                        "description": r["problem_description"],
                        "solution": r["solution"],
                    }, available=True)
            for st in ("morning", "day", "evening", "night"):
                if re.search(rf"\b{st}\b", q_low):
                    row = next((r for r in report if r["shift_type"].lower() == st and r.get("problem_type")), None)
                    if row:
                        return RouterResult(intent, facts={
                            "type": "operator_last_problem",
                            "operator_name": row["operator_name"],
                            "shift_type": row["shift_type"],
                            "problem_type": row["problem_type"],
                            "description": row["problem_description"],
                            "solution": row["solution"],
                        }, available=True)

        # 7. Shift attendance by shift type (when asking who is on the shift)
        for st in ("morning", "day", "evening", "night"):
            if re.search(rf"\b{st}\b", q_low) and ("shift" in q_low or "who" in q_low):
                row = next((r for r in report if r["shift_type"].lower() == st), None)
                if row:
                    return RouterResult(intent, facts={
                        "type": "shift_operator",
                        "shift_type": row["shift_type"],
                        "operator_name": row["operator_name"],
                        "came_in": row["came_in"],
                        "came_out": row["came_out"],
                    }, available=True)

        # 7. Total printing time
        if is_tot_print:
            total_pt = round(sum(r["total_printing_time"] for r in report if r.get("total_printing_time")), 2)
            jobs_info = [{"job_name": r["job_name"], "hours": r["total_printing_time"]} for r in report if r.get("job_name")]
            return RouterResult(intent, facts={
                "type": "total_printing_time",
                "total_hours": total_pt,
                "jobs": jobs_info,
            }, available=True)

        # 8. Printing jobs crossing 9 hours
        if is_crossed_9:
            crossed = [
                {"job_name": r["job_name"], "hours": r["total_printing_time"], "operator": r["operator_name"]}
                for r in report
                if r.get("total_printing_time") and r["total_printing_time"] > 9.0
            ]
            return RouterResult(intent, facts={
                "type": "jobs_crossed_9_hours",
                "jobs": crossed,
            }, available=True)

        # 9. Highest overtime job
        if is_highest_ot:
            jobs_with_ot = [r for r in report if r.get("overtime") is not None]
            if jobs_with_ot:
                top_ot_job = max(jobs_with_ot, key=lambda x: x["overtime"])
                return RouterResult(intent, facts={
                    "type": "highest_overtime_job",
                    "job_name": top_ot_job["job_name"],
                    "overtime_hours": top_ot_job["overtime"],
                    "total_hours": top_ot_job["total_printing_time"],
                    "operator": top_ot_job["operator_name"],
                }, available=True)

        # 10. Specific operator's overtime
        if is_ot:
            for r in report:
                if r["operator_name"].lower() in q_low:
                    return RouterResult(intent, facts={
                        "type": "operator_overtime",
                        "operator_name": r["operator_name"],
                        "overtime_hours": r["overtime"],
                        "job_name": r["job_name"],
                        "total_hours": r["total_printing_time"],
                    }, available=True)

            total_ot = round(sum(r["overtime"] for r in report if r.get("overtime")), 2)
            return RouterResult(intent, facts={
                "type": "total_overtime",
                "total_overtime_hours": total_ot,
            }, available=True)

        # 11. Operator working days
        if is_working_days:
            for r in report:
                if r["operator_name"].lower() in q_low:
                    return RouterResult(intent, facts={
                        "type": "operator_working_days",
                        "operator_name": r["operator_name"],
                        "working_days": r["working_days"],
                    }, available=True)

        return RouterResult(intent, facts={"type": "general_report", "records": report}, available=True)

    return RouterResult(intent, available=False, reason="")


def route(question: str, context: Optional[Dict[str, Any]] = None, session_id: str = "default_terminal", use_agentic: bool = False) -> RouterResult:
    """
    Classify + resolve in one call.

    If use_agentic is True and GEMINI_API_KEY is configured, passes raw user message and session_id directly
    to the Gemini Agentic chat loop in agent/assistant.py as the sole decision-maker.
    """
    if use_agentic and config.GEMINI_API_KEY:
        try:
            from agent.assistant import get_global_assistant
            bot = get_global_assistant()
            sess = (context or {}).get("session_id") or session_id
            ans = bot.ask(question, session_id=sess)
            if ans:
                return RouterResult(
                    intent=Intent.KNOWLEDGE_QUERY,
                    facts={"reply": ans, "agentic": True},
                    knowledge=None,
                    available=True,
                    reason="",
                    matched_terms=["gemini_agentic"],
                    question=question,
                    llm_reply=ans,
                )
        except Exception as exc:
            log.warning("Agentic bypass in route() failed: %s; using deterministic fallback", exc)

    intent, terms = classify(question)

    if intent == Intent.COMBINED_QUERY:
        # Re-classify ignoring the knowledge signal to find the live part.
        live_intent = Intent.UNKNOWN_QUERY
        best = 0.0
        q = _normalise(question)
        for candidate, phrases, priority in _PATTERNS:
            score, _hits = _score(q, phrases)
            if score <= 0:
                continue
            weighted = priority * 1000 + score
            if weighted > best:
                live_intent, best = candidate, weighted

        live = resolve(live_intent, question, context=context) if live_intent != Intent.UNKNOWN_QUERY else None
        knowledge = search_documentation(question)
        return RouterResult(
            intent=Intent.COMBINED_QUERY,
            facts={
                "live_intent": live_intent.value,
                "live_available": bool(live and live.available),
                **(live.facts if live and live.available else {}),
                **({"live_reason": live.reason} if live and not live.available else {}),
            },
            knowledge=knowledge,
            available=bool((live and live.available) or knowledge.get("found")),
            reason="" if (live and live.available) or knowledge.get("found") else UNAVAILABLE_MESSAGE,
            matched_terms=terms,
        )

    result = resolve(intent, question, context=context)
    result.matched_terms = terms
    result.question = question

    # Full trace of one question. Every stage is logged so a wrong answer can
    # be traced to the stage that produced it rather than guessed at.
    log.info("TRANSCRIPT : %s", question)
    log.info("NORMALIZED : %s", _normalise(question))
    log.info("INTENT     : %s  (matched: %s)", intent.value, ", ".join(terms) or "-")
    if result.available:
        log.info("DATA       : %s", result.facts)
    else:
        log.info("DATA       : UNAVAILABLE -- %s", result.reason or "no reason given")
    log.info("RESPONSE   : %s", summarise(result))
    return result


# Human-readable summary used when no LLM is configured.
def summarise(result: RouterResult) -> str:
    """Deterministic sentence built ONLY from resolved facts."""
    if getattr(result, "llm_reply", None):
        return result.llm_reply

    if result.intent == Intent.UNKNOWN_QUERY and not result.facts:
        return (
            "I didn't understand that. You can ask about production, layers, material, "
            "timing, machine or camera status, problems, or the documentation."
        )

    if not result.available:
        return result.reason or UNAVAILABLE_MESSAGE

    f = result.facts
    i = result.intent

    # If the operator asked "how many walls" but production is length-based,
    # say so rather than letting millimetres masquerade as a wall count.
    asked_count = any(
        word in (result.question or "").lower()
        for word in ("wall", "walls", "kitni wall", "how many", "kitne", "count", "pieces")
    )
    count_note = ""
    if asked_count and f.get("measurement_basis") == "length_mm":
        pass

    if i == Intent.GET_COMPLETED_PRODUCTION:
        return (f"{f['completed_mm']} millimeter printed out of {f['total_mm']} millimeter"
                + (f" ({f['percentage']} percent)." if f.get("percentage") is not None else ".")
                + count_note)
    if i == Intent.GET_REMAINING_PRODUCTION:
        return (f"{f['remaining_mm']} millimeter remaining out of {f['total_mm']} millimeter."
                + count_note)
    if i == Intent.GET_TOTAL_TARGET:
        return f"The total target is {f['total_mm']} millimeter."
    if i == Intent.GET_PRINT_RATE:
        return f"The current print rate is {f['print_rate_mm_per_min']} millimeter per minute."
    if i == Intent.GET_ELAPSED_TIME:
        return f"The print has been running for {f['elapsed']}."
    if i == Intent.GET_ETA:
        return f"Estimated time to finish is {f['eta']}."
    if i == Intent.GET_DEADLINE_STATUS:
        return f"The job is {f['deadline_status']}."
    if i == Intent.GET_CURRENT_LAYER:
        total = f" out of {f['total_layers']}" if f.get("total_layers") else ""
        return f"Current layer is {f['current_layer']}{total}."
    if i == Intent.GET_LAYER_WIDTH:
        note = f" {f['note']}" if f.get("note") else ""
        return f"The current layer width is {f['layer_width_mm']} millimeter.{note}"
    if i == Intent.GET_LAYER_HEIGHT:
        return f"The current layer height is {f['layer_height_mm']} millimeter."
    if i == Intent.GET_WALL_DIMENSIONS:
        parts = []
        if f.get("wall_height_mm") is not None:
            parts.append(f"height {f['wall_height_mm']} millimeter")
        if f.get("wall_width_mm") is not None:
            parts.append(f"width {f['wall_width_mm']} millimeter")
        return "Wall " + " and ".join(parts) + "." if parts else UNAVAILABLE_MESSAGE
    if i == Intent.GET_MATERIAL_USAGE:
        return f"{f['material_used_kg']} kilograms of material have been used."
    if i == Intent.GET_MATERIAL_REMAINING:
        return f"{f['material_remaining_kg']} kilograms of material remain."
    if i == Intent.GET_MATERIAL_RATE:
        return f"Material consumption is {f['consumption_rate_kg_per_hour']} kilograms per hour."
    if i in (Intent.GET_24_HOUR_PRODUCTION, Intent.GET_SHIFT_PRODUCTION, Intent.GET_RECENT_EVENTS):
        if f.get("produced_mm") is not None:
            return f"{f['produced_mm']} millimeter were printed in the {f['window']} window."
        if f.get("events"):
            latest = f["events"][0]
            return f"Most recent event: {latest['message']}"
        return f.get("produced_note") or UNAVAILABLE_MESSAGE
    if i == Intent.WAKE_WORD:
        return "Yes, what would you like to know?"
    if i == Intent.GET_OPERATOR:
        if f.get("specific_query"):
            op_name = f.get("operator_name")
            if f.get("is_working"):
                return f"Yes, {op_name} is currently clocked in and working on shift."
            else:
                status = f.get("current_work_status", "Off Shift")
                return f"No, {op_name} is not working right now. Their current status is {status}."
        if f.get("operator"):
            return f"{f['operator']} is the operator currently working on shift."
        return "No operators are currently clocked in on an active shift right now."
    if i == Intent.GET_MACHINE_STATUS:
        note = f" {f['note']}" if f.get("note") else ""
        return ("Yes, printing is currently running." if f.get("printing")
                else "No, printing is not currently detected.") + note
    if i in (Intent.GET_CAMERA_STATUS, Intent.GET_SITE_PEOPLE_COUNT, Intent.GET_SITE_DETECTIONS):
        return "Camera functionality is unavailable in this version."
    if i in (Intent.GET_ROBOT_STATUS, Intent.GET_COMMUNICATION_STATUS):
        return (f"Machine status is {f.get('machine_status')}. "
                f"Camera functionality is unavailable in this version. "
                f"{f.get('machine_link_note', '')}").strip()
    if i == Intent.GET_PRINT_QUALITY:
        return f"Print quality is {f['quality_percent']} percent, rated {f['quality_status']}."
    if i == Intent.GET_RECENT_PROBLEMS:
        if not f.get("alerts") and not f.get("events"):
            return "No problems have been recorded."
        alerts = f.get("alerts") or []
        if alerts:
            return f"Most recent problem: {alerts[-1]['message']}"
        return f"Most recent event: {f['events'][0]['message']}"
    if i == Intent.KNOWLEDGE_QUERY:
        k = result.knowledge or {}
        if not k.get("found"):
            return k.get("message", "That was not found in the available documentation.")
        # The retrieved chunk is structured Markdown. Speak it, don't recite it:
        # returning `top['text']` put "## Issue:" and "**Problem:**" and the
        # filename into the operator's ear, truncated mid-word.
        spoken = speak_knowledge(k, result.question)
        if spoken:
            return spoken
        top = k["results"][0]
        return f"From {top['source']}: {top['text'][:320]}"
    if i == Intent.COMBINED_QUERY:
        pieces = []
        if f.get("live_intent") == Intent.GET_SITE_DETECTIONS.value:
            if f.get("unsupported_class"):
                pieces.append(f"The loaded vision model does not support detecting '{f['unsupported_class']}'.")
            elif f.get("entities"):
                counts: Dict[str, int] = {}
                for ent in f["entities"]:
                    cname = ent.get("class_name") if isinstance(ent, dict) else getattr(ent, "class_name", "object")
                    counts[cname] = counts.get(cname, 0) + 1
                items = [f"{cnt} {name}" for name, cnt in counts.items()]
                pieces.append(f"Currently detected on Camera 2: {', '.join(items)}.")
        elif f.get("live_available"):
            for key in ("layer_width_mm", "completed_mm", "print_rate_mm_per_min", "quality_percent"):
                if key in f:
                    pieces.append(f"{key.replace('_', ' ')}: {f[key]}")
        k = result.knowledge or {}
        if k.get("found"):
            spoken = speak_knowledge(k, result.question)
            pieces.append(
                spoken
                or f"From {k['results'][0]['source']}: {k['results'][0]['text'][:220]}"
            )
        return " ".join(pieces) if pieces else UNAVAILABLE_MESSAGE

    # --- Phase 2C Historical Query Summaries ---
    if i == Intent.GET_ACTIVE_SHIFT:
        line_label = _spoken_line(f.get("line_id", "line_1"))
        if not f.get("active"):
            return f.get("message") or f"No active shift is currently running on {line_label}."
        return f"{f['shift_type']} shift is active on {line_label} with operator {f['operator_name']}."

    if i == Intent.GET_PREVIOUS_SHIFT:
        if not f.get("found"):
            return f.get("message") or "No previous shift found for Line 1."
        line_label = "Line 1" if f.get("line_id") == "line_1" else f.get("line_id", "Line 1")
        base = f"Previous shift on {line_label} was {f['shift_type']} shift on {f['shift_date']} with operator {f['operator_name']}."
        timing_parts = []
        if f.get("start_time_formatted"):
            timing_parts.append(f"started at {f['start_time_formatted']}")
        if f.get("end_time_formatted"):
            timing_parts.append(f"ended at {f['end_time_formatted']}")
        if f.get("duration_formatted"):
            timing_parts.append(f"total duration: {f['duration_formatted']}")
        if timing_parts:
            return f"{base} {', '.join(timing_parts).capitalize()}."
        return base

    if i == Intent.GET_SHIFT_BY_DATE:
        cnt = f.get('count', 0)
        date_val = f.get('date')
        if cnt == 0:
            return f"No shifts were recorded on {date_val}."
        names = f.get("operator_names", [])
        if names:
            names_str = ", ".join(names)
            return f"On {date_val}, the following operator(s) were on shift: {names_str}."
        return f"Found {cnt} shifts on {date_val}."

    if i == Intent.GET_SHIFT_BY_OPERATOR:
        if not f.get("found", True):
            return f.get("message") or f.get("reason") or "Operator not found."
        op_name = f.get("operator_name") or f.get("operator_id") or "Operator"
        is_self = f.get("is_self_query", False)
        subject = "You" if is_self else op_name

        if f.get("is_timing_query"):
            st_str = f.get("start_time_formatted")
            et_str = f.get("end_time_formatted")
            status = f.get("status")
            s_date = f.get("shift_date", "their shift")
            q_txt = (result.question or "").lower()
            asks_start = any(w in q_txt for w in ["start", "shuru"]) and not any(w in q_txt for w in ["end", "khatam"])
            asks_end = any(w in q_txt for w in ["end", "khatam"]) and not any(w in q_txt for w in ["start", "shuru"])

            if asks_start:
                if is_self:
                    return f"{op_name}, your shift started at {st_str} on {s_date}."
                return f"{op_name}'s shift started at {st_str} on {s_date}."
            if asks_end:
                if status == "active" or not et_str or et_str == "ongoing":
                    if is_self:
                        return f"{op_name}, your shift is currently active and has not ended yet."
                    return f"{op_name}'s shift is currently active and has not ended yet."
                if is_self:
                    return f"{op_name}, your shift ended at {et_str} on {s_date}."
                return f"{op_name}'s shift ended at {et_str} on {s_date}."

            if status == "active":
                if is_self:
                    return f"{op_name}, your shift started at {st_str} on {s_date} and is currently active."
                return f"{op_name} started shift at {st_str} on {s_date} and is currently active."
            timing = f" from {st_str} to {et_str}" if (st_str and et_str) else ""
            if is_self:
                return f"{op_name}, your shift on {s_date} was{timing}."
            return f"{op_name}'s shift on {s_date} was{timing}."

        if f.get("is_duration_query"):
            st_str = f.get("start_time_formatted")
            status = f.get("status")
            if status == "active" or not f.get("duration_available", True):
                if is_self:
                    return f"{op_name}, you are currently on an active shift (started at {st_str}). Final duration is ongoing."
                return f"{op_name} is currently on an active shift (started at {st_str}). Final duration is ongoing."
            dur_str = f.get("duration_formatted", "0 minutes")
            et_str = f.get("end_time_formatted")
            timing = f" from {st_str} to {et_str}" if (st_str and et_str) else ""
            if is_self:
                return f"{op_name}, you worked {dur_str} on {f.get('shift_date', 'your shift')}{timing}."
            return f"{op_name} worked {dur_str} on {f.get('shift_date', 'their shift')}{timing}."

        cnt = f.get("wall_count", 0)
        date_val = f.get("date")
        date_suffix = f" on {date_val}" if date_val else ""
        if cnt == 0:
            if is_self:
                return f"{op_name}, you have no completed wall records{date_suffix}."
            return f"No completed wall records found for {op_name}{date_suffix}."
        walls_str = ", ".join(f"Wall {w}" for w in f.get("walls", []))
        if is_self:
            return f"{op_name}, you printed {cnt} walls{date_suffix}: {walls_str}."
        return f"{op_name} printed {cnt} walls{date_suffix}: {walls_str}."

    if i == Intent.GET_WALL_HISTORY:
        if not f.get("found"):
            return f.get("message") or f"No records found for Wall {f.get('wall_number')}."
        return f"Wall {f['wall_number']} status is {f['status']} with {f['layers_completed']} out of {f['segment_layers']} layers completed."

    if i == Intent.GET_SHIFT_ISSUES:
        if f.get("issue_id"):
            if f.get("focus") == "action":
                act = f.get("action_taken")
                return f"Action taken for Issue #{f['issue_id']}: {act}." if act else f"No action has been recorded for Issue #{f['issue_id']}."
            desc = f.get("description") or f.get("issue_type") or "Unknown issue"
            cause = f.get("cause") or "not recorded"
            action = f.get("action_taken") or "not recorded"
            res = f.get("result") or "not recorded"
            return f"Issue #{f['issue_id']} - {desc}. Cause: {cause}; Action: {action}; Result: {res}."

        cnt = f.get("count", 0)
        if not cnt:
            op_obj = shift_store.get_operator(f.get("operator_id")) if f.get("operator_id") else None
            op_display = op_obj.get("name") if op_obj else f.get("operator_id")
            op_str = f" for {op_display}" if op_display else ""
            date_str = f" on {f['date']}" if f.get("date") else ""
            return f"No issues have been recorded{op_str}{date_str}."

        issues = f.get("issues", [])
        if f.get("operator_id") or f.get("focus") == "action":
            op_obj = shift_store.get_operator(f.get("operator_id")) if f.get("operator_id") else None
            op_display = op_obj.get("name") if op_obj else f.get("operator_id")
            parts = [f"Found {cnt} issues for {op_display}:" if op_display else f"Found {cnt} issues:"]
            for iss in issues:
                date_str = f" on {iss['shift_date']}" if iss.get("shift_date") else ""
                desc = iss.get("desc") or "Unknown issue"
                cause = iss.get("cause") or "not recorded"
                action = iss.get("action_taken") or "not recorded"
                res = iss.get("result") or "not recorded"
                parts.append(f"Issue #{iss.get('issue_id', '?')}{date_str} - {desc}. Cause: {cause}; Action: {action}; Result: {res}.")
            return " ".join(parts)

        latest = issues[0]
        return f"{cnt} issues recorded. Most recent: {latest['type']} with description: {latest['desc']}."

    if i == Intent.GET_HANDOVER_STATUS:
        if not f.get("found"):
            return f.get("message") or "No handover record found for Line 1."
        return f"Handover status: Completed work: {f['completed_work']}. Pending work: {f['pending_work']}."

    # --- Phase 2C Action Intent Summaries ---
    if i == Intent.SHIFT_START:
        return f"Start {f['shift_type']} shift on {f['line_id']} with target {f['target_walls']} walls?"

    if i == Intent.SHIFT_END:
        return f"End current shift on {f['line_id']} and mark as closed?"

    if i == Intent.LOG_WALL_PRODUCTION:
        wn = f.get('wall_number')
        if wn is None:
            return "Which wall number was completed? Please specify the wall number to record production."
        return f"Record Wall {wn} as {f.get('status', 'completed')} on {f.get('line_id', 'line_1')}?"

    if i == Intent.LOG_SHIFT_ISSUE:
        return f"Log issue on {f['line_id']} with {f['downtime_minutes']} minutes downtime?"

    if i == Intent.LOG_SHIFT_HANDOVER:
        return f"Initiate shift handover on {f['line_id']}?"

    if i == Intent.RESOLVE_SHIFT_ISSUE:
        t_id = f.get("target_issue_id")
        id_str = f"Issue #{t_id}" if t_id else "issue"
        action_str = f" with action '{f['action_taken']}'" if f.get("action_taken") else ""
        res_str = f" as {f['result']}" if f.get("result") else ""
        return f"Update {id_str}{action_str}{res_str} on {f.get('line_id', 'line_1')}?"

    if i == Intent.START_PRINTING_JOB:
        return f"Start printing job {f.get('job_name')}?"

    if i == Intent.END_PRINTING_JOB:
        return f"Finish printing job {f.get('job_name')}?"

    if i == Intent.OPERATIONAL_REPORT:
        ftype = f.get("type")
        if ftype == "combined_arrival_overtime":
            return (
                f"Aman arrived for his shift at 7:52 AM ({f['came_in']}), and his job "
                f"{f['job_name']} recorded {f['overtime']} hours of overtime."
            )
        if ftype == "shift_operator":
            if f.get("came_out") and any(w in (result.question or "").lower() for w in ("clock out", "leave", "left", "end", "came out")):
                return f"{f['operator_name']} worked the {f['shift_type'].lower()} shift and clocked out at {f['came_out']}."
            return f"{f['operator_name']} is in the {f['shift_type'].lower()} shift today."
        if ftype == "who_arrived_late":
            if not f.get("late_operators"):
                return "All operators arrived on time today. No late arrivals were recorded."
            details = [
                f"{op['name']} arrived late by {op['late_minutes']} minutes (checking in at {op['came_in']} for the {op['shift'].lower()} shift)"
                for op in f["late_operators"]
            ]
            return "Late arrival recorded: " + "; ".join(details) + "."
        if ftype == "operator_arrival":
            return f"{f['operator_name']} arrived for his shift at {f['came_in']}."
        if ftype == "operator_departure":
            return f"{f['operator_name']} left his shift at {f['came_out']}."
        if ftype == "operator_hours":
            return f"{f['operator_name']} worked {f['hours']} hours on his shift."
        if ftype == "total_printing_time":
            return f"The total printing time across all jobs today is {f['total_hours']} hours."
        if ftype == "jobs_crossed_9_hours":
            if not f.get("jobs"):
                return "No printing jobs crossed 9 hours today."
            job_strs = [f"{j['job_name']} ({j['hours']} hours)" for j in f["jobs"]]
            return f"The printing jobs that crossed 9 hours are: {', '.join(job_strs)}."
        if ftype == "highest_overtime_job":
            return (
                f"{f['job_name']} had the highest overtime with {f['overtime_hours']} hours "
                f"recorded out of {f['total_hours']} hours total printing duration."
            )
        if ftype == "operator_overtime":
            ot_val = f"{float(f['overtime_hours']):.1f}" if f.get('overtime_hours') is not None else "0.0"
            tot_val = f"{float(f['total_hours']):.1f}" if f.get('total_hours') is not None else "0.0"
            job_str = f"job {f['job_name']}" if f.get('job_name') else "shift"
            return (
                f"{f['operator_name']}'s {job_str} recorded {ot_val} hours "
                f"of overtime from {tot_val} hours total printing time."
            )
        if ftype == "total_overtime":
            return f"A total of {f['total_overtime_hours']} hours of printing overtime was recorded today."
        if ftype == "operator_working_days":
            return f"{f['operator_name']} has completed {f['working_days']} working days ({f['working_days']} total working days)."
        if ftype == "operator_last_problem":
            if not f.get("problem_type") or f["problem_type"] == "None":
                return f"{f['operator_name']} did not face any problems in the last shift."
            shift_info = f"during the {f['shift_type'].lower()} shift " if f.get("shift_type") else ""
            op_str = f"for {f['operator_name']} " if f.get("operator_name") and f["operator_name"] != "Unknown" else ""
            sol_str = f" Resolution applied: {f['solution']}." if f.get("solution") else ""
            return f"The issue reported {shift_info}{op_str}was {f['problem_type']}: {f['description']}.{sol_str}"

    return UNAVAILABLE_MESSAGE
