# Kelvin6K Conflicting Values - MUST CONFIRM, DO NOT ASSUME

These are places where the source documents disagree, or where a value is
context-specific and has been mistaken for a standard. None of them is resolved
here. Kelvin6K_Printing_Data_Review_Problems_Solutions.pdf sections 3.3, 3.4,
3.5 and 3.6 marks these as MUST CONFIRM, CONTEXT-SPECIFIC or OBSERVATION ONLY.

When a question touches any of these, the assistant must state both values and
ask which print or setting is meant. It must NOT pick one.

## Issue: Layer Thickness - Conflicting Values 40 mm and 50 mm

**Problem:** The available data contains different layer-thickness values in different contexts. No universal production standard is established.
**Symptoms / Observation:** Layers reported as too thick in one context; U-joints between walls failing to join in another.
**Cause:** Context-dependent. In the U-joint case, 40 mm overlapped the adjacent layer. In the separate too-thick case, 40 mm was reported alongside a 120 mm distance.
**Solution / Corrective Action (CONFLICT - DO NOT ASSUME A STANDARD):** The dataset contains both 40 mm and 50 mm values in different contexts. Confirm the standard value for the active print and settings before changing anything.
  - U-joint case: 40 mm was changed to 50 mm because 40 mm overlapped the adjacent layer. Another message suggested plus 15 mm, subject to checking thickness and speed. Verify actual joint fit before repeating.
  - Too-thick case: reduce layer thickness; the operator reported 40 mm thickness and 120 mm distance. Use the verified setting for that print.
**Important Parameters:** 40 mm, 50 mm, plus 15 mm suggested, 120 mm distance. CONFLICTING - none confirmed as the standard.
**Safety / Important Notes:** Do not state either value as the universal standard. Ask which print and which joint before advising.
**English Questions:** What layer thickness should I use? Is 40 mm or 50 mm correct? What is the standard layer thickness? Layer thickness 40 vs 50 mm.
**Hinglish Questions:** Layer thickness kitni honi chahiye? 40 ya 50 mm? Standard thickness kya hai?
**Source:** PDF 4 records 26 and 29; flagged in PDF 1 section 3.3.
**Verification Status:** CONFLICT - MUST CONFIRM WITH OPERATOR. Both values are site-reported in different contexts. Neither is confirmed as standard.

## Issue: Fly Ash Fill Depth - Conflicting Values 3 Layers and 5 Layers

**Problem:** Two site records give different fly ash fill depths.
**Symptoms / Observation:** Fly ash used for bottom-layer filling and defect correction, with different depths recorded.
**Cause:** Context-dependent, or a change in practice between the two records. Not resolved in the sources.
**Solution / Corrective Action (CONFLICT - DO NOT ASSUME):** Both instructions appear in the site records:
  - Fill the bottom three layers with fly ash, and add fly ash around layers where instructed.
  - Fill fly ash up to five layers inside and outside as instructed.
  Confirm which applies to the active wall before filling.
**Important Parameters:** 3 layers and 5 layers. CONFLICTING - neither confirmed as standard.
**Safety / Important Notes:** Both records say "as instructed", suggesting the depth is decided per wall rather than fixed.
**English Questions:** How many layers of fly ash? Do I fill three layers or five layers with fly ash? Fly ash depth.
**Hinglish Questions:** Fly ash kitni layers tak bharna hai? Teen ya paanch layers? Fly ash kahan dalna hai?
**Source:** PDF 4 records 29 and 30.
**Verification Status:** CONFLICT - MUST CONFIRM WITH OPERATOR.

## Issue: Feedrate - 150 Percent Meaning Two Different Things

**Problem:** The value 150 percent appears in two documents with two different meanings, and 170 percent was rejected.
**Symptoms / Observation:** Elevation print quality unacceptable at 170 percent.
**Cause:** Not a fault - a documentation ambiguity.
**Solution / Corrective Action (CONFLICT - DO NOT ASSUME):**
  - 170 percent was reported unacceptable for elevation printing; reduce speed and accelerator ratio.
  - 150 percent was recorded in a later update after that correction.
  - 150 percent also appears separately as the Feedrate Override reading in a machine screen snapshot, which is simply what the machine happened to be set to at that moment.
  The sources do not establish that 150 percent is the formal maximum or a universal setting. Confirm the approved setting for the active print.
**Important Parameters:** 170 percent rejected for elevation. 150 percent recorded after correction. 150 percent also observed as a live snapshot reading. Related recorded value 3000 mm/min during arc troubleshooting. Machine limits observed: Max Velocity 6000, Max Rapid 6000, Current Feedrate 3600.
**Safety / Important Notes:** Quality takes priority over speed for elevation walls.
**English Questions:** What feedrate should I use? Is 150 percent the maximum? Is 170 percent too fast? What is the approved speed?
**Hinglish Questions:** Feedrate kitna rakhna hai? 150 percent maximum hai? Speed kitni set karni hai?
**Source:** PDF 4 record 11; PDF 2 section 2; flagged in PDF 1 section 3.6.
**Verification Status:** CONFLICT - MUST CONFIRM. 170 percent rejection is site-verified; the approved standard is NOT CONFIRMED.

## Issue: Z-Height Correction - Multiple Recorded Approaches

**Problem:** The source records more than one approach to the same Z-height problem, and disagrees about whether it is a Z-offset or a G-code issue.
**Symptoms / Observation:** A print stopped with approximately 80 mm remaining because the Z-height limit was not set correctly.
**Cause:** Recorded as linked to removal of the Z motor. One message identifies it as a Z-offset issue rather than necessarily a G-code error.
**Solution / Corrective Action (CONFLICT - DO NOT ASSUME):** The source records these, without stating which is correct:
  - Check the Z-height during calibration and manually correct the relevant file or J2 height, then resume.
  - G92.1 followed by G92 Z0 at the current position.
  - Setting a suitable Z value such as G92 Z1515.
**Important Parameters:** Approximately 80 mm remaining. G92.1, G92 Z0, G92 Z1515. CONFLICTING approaches.
**Safety / Important Notes:** MACHINE-SPECIFIC PROCEDURE - AUTHORIZATION REQUIRED. These are controlled procedures, not generic troubleshooting instructions. Do not issue any of these commands without authorization and without identifying the exact machine condition.
**English Questions:** How do I correct the Z height? Which G92 command should I use? Is this a Z-offset or a G-code problem?
**Hinglish Questions:** Z height kaise theek karein? Kaunsa G92 command use karna hai?
**Source:** PDF 4 record 25; flagged in PDF 1 section 3.8.
**Verification Status:** CONFLICT within a single site record. AUTHORIZATION REQUIRED.

## Issue: Fallen Wall Cause - Sand Compaction or Wet Layers

**Problem:** Two records attribute a fallen wall to different causes.
**Symptoms / Observation:** A printed wall fell. A related base defect was attributed to bottom sand compaction.
**Cause (DISPUTED - NEITHER CONFIRMED):**
  - One record attributes a defect to insufficient attention during bottom sand compaction.
  - A later assessment says the observed pattern looked more consistent with wet layers than with sand compression.
  Neither is a confirmed root cause.
**Solution / Corrective Action:** Both records agree on the checks: inspect CCTV and layer condition; ensure layers are sufficiently dry before proceeding; and prepare the base correctly by levelling the sand surface and placing polythene before printing.
**Important Parameters:** None numeric.
**Safety / Important Notes:** Do not state either cause as confirmed. Investigate wall instability with CCTV and physical inspection.
**English Questions:** Why did the wall fall? Was it the sand or the wet layers? What causes a wall to collapse?
**Hinglish Questions:** Wall kyun gir gayi? Sand ki galti thi ya layers geeli thi? Wall collapse ka kya reason hai?
**Source:** PDF 4 records 30 and 31.
**Verification Status:** CONFLICT - CAUSE DISPUTED, neither confirmed.

## Issue: Total Layer Count 600 Is Not a Standard

**Problem:** An observed total of 600 layers has been mistaken for a universal wall layer count.
**Symptoms / Observation:** The machine screen showed 115 completed layers plus 485 remaining, totalling 600.
**Cause:** Not applicable - this is a context-specific reading.
**Solution / Corrective Action:** Treat 600 as belonging to that specific print job only. The arithmetic is internally consistent, but it must not be treated as a universal wall layer count. Read the layer count for the active job from live machine telemetry, which is not yet connected to Kelvin6K.
**Important Parameters:** 115 completed, 485 remaining, 600 total - for one specific print job only. Related: the observed layer time of approximately 1 minute 30 seconds is a SINGLE OBSERVATION and not a standard layer time.
**Safety / Important Notes:** Kelvin6K's database currently holds a different total-layer value for its seeded job. Do not reconcile the two by assumption.
**English Questions:** How many layers are in a wall? Is 600 layers standard? What is the total layer count?
**Hinglish Questions:** Ek wall mein kitni layers hoti hain? 600 layers standard hai? Total layers kitni hain?
**Source:** PDF 2 section 2; flagged in PDF 1 sections 3.4 and 3.5.
**Verification Status:** CONTEXT-SPECIFIC VALUE - NOT a standard. MUST CONFIRM per job.
