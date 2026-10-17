# Kelvin6K Operational Rules and Maintenance Intervals

Sources: 3D_Printing_Issues_Solutions_AI_Agent_Dataset_No_Atom_V2(3).pdf
("Operational Rules Extracted from the Chat") and
Kelvin6K_Printing_Issues_and_Solutions_Dataset(2).pdf section 3
("Agent Response Logic").

## Issue: Cleaning Intervals - Accelerator Hole, Pipe, Hopper and Pump Front

**Problem:** Four different parts are cleaned on four different intervals, counted in different units. They sound similar and are easily confused.
**Symptoms / Observation:** Material flow becomes less reliable; print quality drifts; a maintenance reminder appears.
**Cause:** Overdue routine cleaning.
**Solution / Corrective Action:**
  - Accelerator hole or pipe - clean every 5 LAYERS, using the big Allen key or an appropriate rod.
  - Printing pipe - clean every 4 HOURS of operation.
  - Hopper - clean after EACH WALL.
  - Front of the pump - clean after every 2 WALLS.
**Important Parameters:** 5 layers - accelerator hole. 4 hours - printing pipe. 1 wall - hopper. 2 walls - pump front.
**Safety / Important Notes:** Do NOT use wire to clean the accelerator hole. If the machine must be stopped for that cleaning, stop in the column area and keep the stop under five seconds. The accelerator HOLE (5 layers, Allen key) and the printing PIPE (4 hours) are different tasks with different tools - do not substitute one for the other.
**English Questions:** How often do I clean the accelerator hole? How often do I clean the printing pipe? When do I clean the hopper? When do I clean the front of the pump? What cleaning is due?
**Hinglish Questions:** Accelerator hole kitni layers ke baad clean karna hai? Pipe kitne ghante mein clean karna hai? Hopper kab clean karna hai?
**Source:** PDF 4 records 34, 49; PDF 3 PRINT-021.
**Verification Status:** SITE-VERIFIED for the layer and wall intervals; the 4-hour pipe interval is DOCUMENTED and consistent with the operating manual.

## Issue: Inspection and Timing Intervals - Coupling, Seals, Pump Running, Restart and Curing

**Problem:** Operators need to know the time-based inspection and operating intervals.
**Symptoms / Observation:** Overdue inspection; material hardening during pauses; long restart gaps; walls not cured.
**Cause:** Overdue inspection or an interval not being observed.
**Solution / Corrective Action:**
  - Accelerator coupling - inspect DAILY and correct any looseness or wear.
  - Pump seals - inspect WEEKLY and replace or service if worn.
  - Pump during pauses - run every 5 MINUTES to prevent choking and material hardening.
  - Alignment and restart after cleaning - complete within 1 HOUR maximum.
  - Printed walls - cure 3 TIMES PER DAY.
**Important Parameters:** Daily - accelerator coupling. Weekly - pump seals. 5 minutes - pump run during pauses. 1 hour - maximum restart window. 3 times per day - curing.
**Safety / Important Notes:** One hour is repeatedly stated in the source as the maximum restart window. Kelvin6K has no maintenance history table yet, so it can state these intervals but cannot say whether a task is actually due.
**English Questions:** How often are pump seals checked? Has the accelerator coupling been checked today? How often should walls be cured? What is the maximum restart time? How often should the pump run during a pause?
**Hinglish Questions:** Pump seal kab check karna hai? Coupling roz check karni hai? Curing din mein kitni baar karni hai? Restart kitni der mein karna hai?
**Source:** PDF 4 records 14, 39, 49; PDF 3 PRINT-022, PRINT-023.
**Verification Status:** SITE-VERIFIED for the pump-run, restart and curing intervals; DOCUMENTED (project-derived) for the daily and weekly inspections.

## Issue: Pre-Print Verification and Planning Rules

**Problem:** Repeated stoppages and wrong prints were traced to planning and verification gaps rather than to the machine.
**Symptoms / Observation:** Wrong files printed, unclear next wall, missing consumables, no space to print.
**Cause:** Process and planning gaps.
**Solution / Corrective Action:** The site rules for before a print starts:
  - Printing is the primary activity. Avoid using the printing team for unrelated heavy work when site labour can be used.
  - Plan material, labour, hose, sieve and other requirements in advance so printing does not stop unexpectedly.
  - Before printing, verify that the file, wall number, G-code and image all match.
  - Keep wall numbers, print logs, G-code names and panel images synchronised.
  - Screen sand with a sieve or mesh when stone contamination is a risk.
  - Prepare the print base by levelling the sand surface and placing polythene before printing.
  - Plan print locations and print parallel where possible; avoid side prints that block future positions.
**Important Parameters:** A verified sequence of at least 20 prints was requested. Target discussed: normally 2 walls per shift, recorded as discussed rather than confirmed.
**Safety / Important Notes:** Do not start printing without a sieve where required.
**English Questions:** What should I check before printing? How do I prepare the base? What planning is needed before a print? Do I need a sieve?
**Hinglish Questions:** Printing se pehle kya check karna hai? Base kaise taiyaar karna hai? Sieve lagana zaroori hai?
**Source:** PDF 4, Operational Rules Extracted from the Chat.
**Verification Status:** SITE-VERIFIED practice.

## Issue: During-Print, Restart and Handover Rules

**Problem:** Time was lost between walls, and quality was compromised by prioritising speed.
**Symptoms / Observation:** Long restart gaps; defects accepted on elevation walls; unclear handover.
**Cause:** Process gaps during and after printing.
**Solution / Corrective Action:** The site rules for during and after a print:
  - After cleaning, complete alignment and restart the next print within a maximum of one hour.
  - For quality-sensitive and elevation walls, prioritise quality over speed, and correct defects rather than accepting them.
  - Keep printed layers sufficiently dry before continuing. Investigate wall instability with CCTV and physical inspection.
  - Keep the pump shaded and monitor motor heating. Spray the E-motor with fresh water during hot daytime operation as instructed.
  - Document every resolved technical issue with the problem, the detailed solution method and, where possible, a video or log for future troubleshooting.
  - Communicate the next wall clearly to the incoming shift.
**Important Parameters:** 1 hour maximum restart window.
**Safety / Important Notes:** One hour is repeatedly stated in the source as the maximum restart window.
**English Questions:** How long should the restart take? What must I do at shift handover? Should I prioritise speed or quality? What do I record after fixing a problem?
**Hinglish Questions:** Restart kitni der mein karna hai? Shift handover mein kya batana hai? Speed ya quality, kya important hai?
**Source:** PDF 4, Operational Rules Extracted from the Chat.
**Verification Status:** SITE-VERIFIED practice.

## Issue: Diagnostic Response Logic for the Assistant

**Problem:** A reported symptom can match several distinct issues, and answering with the wrong one wastes the operator's time or is unsafe.
**Symptoms / Observation:** An operator reports a symptom without enough detail to identify the issue.
**Cause:** Overlapping symptoms across distinct issues.
**Solution / Corrective Action:** The documented response logic:
  - First identify the operator's symptom - narrow layer, wide layer, separation, shift, low flow, height deviation, machine error, or camera problem.
  - Ask only the minimum follow-up questions needed to distinguish likely causes.
  - For safety-critical issues - a nozzle clog under pressure, a machine error, or a stopped pump or accelerator - instruct the operator to pause or stop the relevant process before troubleshooting.
  - After a corrective action, provide a verification step so the operator can confirm whether the problem is actually resolved.
  - The recommended flow is: operator symptom, then issue candidate, then minimum follow-up questions, then likely causes, and a confirmed cause only when verified.
**Important Parameters:** None.
**Safety / Important Notes:** Layer Gap or Void, Delamination, and Poor Layer Bonding can overlap in symptoms but are not identical. Keep the records distinct rather than merging them.
**English Questions:** How should the assistant diagnose a problem? What questions should be asked first? How do I narrow down a printing fault?
**Hinglish Questions:** Problem kaise pata karein? Pehle kya poochna chahiye?
**Source:** PDF 3 section 3; PDF 1 section 3.2.
**Verification Status:** DOCUMENTED GUIDANCE for the assistant's behaviour.

## Issue: Cause Confidence and Measurement Reporting Rules

**Problem:** Presenting a guess as a confirmed diagnosis, or a camera inference as machine fact, misleads the operator.
**Symptoms / Observation:** A possible cause being stated as the cause; a pixel measurement being reported as millimetres.
**Cause:** Not applicable - these are standing rules.
**Solution / Corrective Action:** The documented reporting rules:
  - Do not present a possible cause as a confirmed diagnosis. Correct wording is "Possible causes include ... The root cause is not confirmed yet."
  - Never tell the operator that a camera measurement is a confirmed machine state when the system is only inferring it from motion.
  - Do not convert pixels to millimetres unless layer-width calibration is configured and valid.
  - If a measurement is low-confidence, ask for physical confirmation instead of presenting it as a definite printing failure.
  - Camera and vision results should not override direct machine telemetry when a reliable machine status link is available.
**Important Parameters:** None.
**Safety / Important Notes:** The data review document marks the possible-versus-confirmed distinction as the most important dataset issue, and as MUST FIX BEFORE PRODUCTION USE.
**English Questions:** Is that the confirmed cause? Can the camera tell me the machine state? Why is the layer width not in millimetres?
**Hinglish Questions:** Ye confirm cause hai? Camera se machine ka status pata chalta hai? Layer width mm mein kyun nahi aa rahi?
**Source:** PDF 1 sections 3.1 and 3.10; PDF 3 section 3.
**Verification Status:** DOCUMENTED GUIDANCE for the assistant's behaviour.
