# Kelvin6K Documented Printing Issues (PRINT Series)

Source: Kelvin6K_Printing_Issues_and_Solutions_Dataset(2).pdf.

IMPORTANT PROVENANCE NOTE: that document states it was generated from the
Kelvin6K project ZIP - the operating manual, troubleshooting guide, print-monitor
logic and alert rules. It is therefore project-derived, NOT operator-verified
site knowledge. Every "Likely Cause" below is a hypothesis. It cannot be used to
validate the system it was derived from.

EXCLUSIONS applied to this file:
- PRINT-008, PRINT-009 and PRINT-010 (high nozzle temperature, low nozzle
  temperature, high bed temperature) are EXCLUDED. This is a concrete printing
  machine with a pump, hopper and accelerator, and has no heated nozzle or
  heated bed. Nozzle temperature appears in none of the 50 real site incident
  records. These entries appear to originate from an FDM-shaped simulator.
- PRINT-001, 002, 003, 004, 005, 006 and 018 are EXCLUDED as duplicates of
  docs/kelvin6k_manual.md and docs/kelvin6k_troubleshooting.md, which remain
  canonical for those topics.

## Issue: Low Material Flow

**Problem:** Low material flow is reported; the print bead becomes narrow or material delivery is inconsistent.
**Symptoms / Observation:** Agent reports low flow; print bead becomes narrow or material delivery is inconsistent.
**Cause (LIKELY - NOT CONFIRMED):** Possible nozzle clog, low material supply, delivery-line issue, pump or seal issue, or air in the line. The root cause is not confirmed.
**Solution / Corrective Action:** Inspect the material supply, the nozzle and the delivery line. Check the pump and seals if the issue continues. Use the nozzle-clog procedure when a blockage is suspected.
**Verification:** Flow returns to the normal operating range and layer width stabilises.
**Important Parameters:** None specified in the source.
**Safety / Important Notes:** Severity WARNING.
**Agent Follow-up:** What is the current material-flow reading? Is the hopper sufficiently filled?
**English Questions:** Material flow is low. The bead is getting narrow. Why is material delivery inconsistent?
**Hinglish Questions:** Material flow drop ho gaya hai. Material kam aa raha hai. Flow inconsistent hai.
**Source:** PDF 3, PRINT-007.
**Verification Status:** DOCUMENTED (project-derived). Cause is LIKELY, not operator-confirmed.

## Issue: Print Height Deviation or Layer Skipping

**Problem:** Actual print height differs from the expected height, and layer pitch may be inconsistent.
**Symptoms / Observation:** Actual print height differs from expected height; layer pitch may be inconsistent.
**Cause (LIKELY - NOT CONFIRMED):** Layer skipping, or mechanical and printing inconsistency causing height drift. Not confirmed.
**Solution / Corrective Action:** Pause and inspect the drive and gantry and the printed layer sequence. Check for mechanical problems and layer deposition consistency before continuing.
**Verification:** Actual height returns within the configured tolerance and layer pitch is consistent.
**Important Parameters:** None specified in the source.
**Safety / Important Notes:** Severity WARNING. See also the site-verified Z-height limit record, which is a different and machine-specific problem.
**Agent Follow-up:** How large is the height deviation, and did it begin after a specific layer?
**English Questions:** The print height is wrong. Layers are being skipped. The height does not match the expected value.
**Hinglish Questions:** Print height expected se zyada ya kam ho rahi hai. Layer skip ho rahi hai. Height galat aa rahi hai.
**Source:** PDF 3, PRINT-011.
**Verification Status:** DOCUMENTED (project-derived). Cause is LIKELY, not operator-confirmed.

## Issue: Machine Error State

**Problem:** Machine telemetry reports an error state.
**Symptoms / Observation:** Machine telemetry reports an error state.
**Cause:** The machine or controller has reported an equipment error.
**Solution / Corrective Action:** Pause the print and inspect the machine according to the reported error. Do not continue blindly while a critical machine error is active.
**Verification:** The machine error clears and the underlying cause has been corrected.
**Important Parameters:** None generic. For specific recorded faults see the site-verified records for E08.5 and the servo drives.
**Safety / Important Notes:** Severity CRITICAL. For critical machine errors, pause or stop the relevant process and inspect the machine before continuing. Machine-specific commands are controlled procedures - AUTHORIZATION REQUIRED.
**Agent Follow-up:** What exact error code or message is the machine reporting?
**English Questions:** The machine is showing an error. What do I do about a machine error? There is a fault on the controller.
**Hinglish Questions:** Machine mein error aa raha hai. Controller error de raha hai. Error aane par kya karein?
**Source:** PDF 3, PRINT-012.
**Verification Status:** DOCUMENTED (project-derived).

## Issue: Pump Stopped

**Problem:** The material pump is reported as stopped while printing is expected.
**Symptoms / Observation:** Material pump is reported as stopped while printing is expected.
**Cause:** The pump has stopped or is not running.
**Solution / Corrective Action:** Stop or hold the print as appropriate and check the pump, the power and control signal, the material path and the pump condition.
**Verification:** The pump runs normally and material flow is restored before resuming production.
**Important Parameters:** None specified in the source.
**Safety / Important Notes:** Severity CRITICAL. See also the site-verified record where no ESP output voltage prevented the pump signal from reaching the pump.
**Agent Follow-up:** Is the pump physically stopped, or is only the telemetry reporting it as stopped?
**English Questions:** The pump has stopped. Why is the pump not running? The pump is reported stopped.
**Hinglish Questions:** Pump band ho gaya hai. Pump chal nahi raha. Pump kyun ruk gaya?
**Source:** PDF 3, PRINT-013.
**Verification Status:** DOCUMENTED (project-derived).

## Issue: Accelerator Stopped

**Problem:** The accelerator is reported as stopped.
**Symptoms / Observation:** Accelerator is reported as stopped.
**Cause:** The accelerator system has stopped or is not receiving the expected run signal.
**Solution / Corrective Action:** Check accelerator operation, coupling and control signal. Resolve the issue before relying on normal material setting and curing behaviour.
**Verification:** The accelerator runs normally and the resulting layers have expected bonding and setting behaviour.
**Important Parameters:** None specified in the source.
**Safety / Important Notes:** Severity CRITICAL.
**Agent Follow-up:** Did the accelerator stop suddenly, and is there an error indication?
**English Questions:** The accelerator has stopped. Why is the accelerator not running? Accelerator stopped during printing.
**Hinglish Questions:** Accelerator band ho gaya hai. Accelerator chal nahi raha. Accelerator kyun ruk gaya?
**Source:** PDF 3, PRINT-014.
**Verification Status:** DOCUMENTED (project-derived).

## Issue: Print Quality Too Low

**Problem:** The overall measured print-quality score falls below the configured minimum.
**Symptoms / Observation:** Overall measured print-quality score falls below the configured minimum.
**Cause (LIKELY - NOT CONFIRMED):** One or more measurable print-quality indicators have degraded. Which one is not identified.
**Solution / Corrective Action:** Inspect the printed wall and check the related measurements - layer width, layer detection, height, material flow and visible defects. Correct the underlying condition before continuing.
**Verification:** The quality score returns above the configured minimum and no critical visual defect remains.
**Important Parameters:** None specified in the source.
**Safety / Important Notes:** Severity WARNING.
**Agent Follow-up:** Which quality indicator changed first? Is the issue visible on the wall?
**English Questions:** The print quality score is low. Why is quality below the minimum? Print quality has dropped.
**Hinglish Questions:** Print quality kharab aa rahi hai. Quality score kam hai. Print acha nahi aa raha.
**Source:** PDF 3, PRINT-015.
**Verification Status:** DOCUMENTED (project-derived). Cause is LIKELY, not operator-confirmed.

## Issue: Layer Missing While Printing

**Problem:** The machine appears to be printing, but the vision system cannot confidently detect a layer.
**Symptoms / Observation:** Machine appears to be printing, but the vision system cannot confidently detect a layer.
**Cause (LIKELY - NOT CONFIRMED):** The layer is absent, obscured or poorly visible, or the vision measurement has insufficient confidence. Not confirmed.
**Solution / Corrective Action:** Check the printed area and the camera view. Confirm lighting and framing and inspect whether material was actually deposited. Do NOT treat a low-confidence measurement as a confirmed physical failure without inspection.
**Verification:** A layer is consistently detected with adequate confidence and the physical print matches the expected deposition.
**Important Parameters:** None specified in the source.
**Safety / Important Notes:** Severity WARNING. A low-confidence camera measurement must not be presented as a confirmed physical failure.
**Agent Follow-up:** Can you physically see material being deposited? Is the camera view clear?
**English Questions:** The system says the layer is missing but the machine is printing. No layer detected. Vision cannot see the layer.
**Hinglish Questions:** Layer detect nahi ho rahi par machine chal rahi hai. Vision system layer nahi dekh raha.
**Source:** PDF 3, PRINT-016.
**Verification Status:** DOCUMENTED (project-derived). Cause is LIKELY, not operator-confirmed.

## Issue: Camera or Vision Error

**Problem:** No usable camera frame or vision analysis is available.
**Symptoms / Observation:** No usable camera frame or vision analysis is available.
**Cause:** The camera is disconnected, unavailable, or unable to provide a usable frame.
**Solution / Corrective Action:** Check the camera connection, the power and USB connection, the camera selection and the camera view. Restore a valid frame before relying on vision-based printing status.
**Verification:** The camera provides stable frames and vision analysis resumes.
**Important Parameters:** None specified in the source.
**Safety / Important Notes:** Severity CRITICAL. This is a vision-system fault, which is distinct from the site-verified record about site camera image quality and network availability.
**Agent Follow-up:** Is the camera connected and showing a live image?
**English Questions:** The camera is not working. No camera frame available. Vision system error.
**Hinglish Questions:** Camera feed nahi aa rahi. Vision system print detect nahi kar raha. Camera connect nahi hai.
**Source:** PDF 3, PRINT-017.
**Verification Status:** DOCUMENTED (project-derived).

## Issue: Printing Status Not Reliable or Jumpy

**Problem:** The agent alternates between printing and not printing, or between layer detected and missing, because of individual bad frames.
**Symptoms / Observation:** Status flips between states on individual frames.
**Cause:** Single-frame noise, low-confidence measurements, or camera motion can cause unstable state decisions.
**Solution / Corrective Action:** Use confidence thresholds and temporal debouncing. Inspect camera stability and the region of interest. The print monitor is designed to wait for consecutive agreeing frames rather than switching on one bad frame.
**Verification:** Status remains stable across multiple consecutive frames during steady printing or idle periods.
**Important Parameters:** None numeric in the source.
**Safety / Important Notes:** Severity WARNING.
**Agent Follow-up:** Does the status change on every frame or only after several consistent observations?
**English Questions:** The printing status keeps changing. Why is the status jumpy? The state flips between printing and not printing.
**Hinglish Questions:** Printing status baar baar badal raha hai. Status stable nahi hai.
**Source:** PDF 3, PRINT-019.
**Verification Status:** DOCUMENTED (project-derived).

## Issue: Printing Activity Not Detected by Camera

**Problem:** The machine is physically moving and printing but the vision monitor reports no printing activity.
**Symptoms / Observation:** Machine is physically moving or printing but the vision monitor reports no printing activity.
**Cause (LIKELY - NOT CONFIRMED):** The motion calculation may be diluted by using the whole frame instead of the print region of interest, or the camera view and region of interest may be wrong.
**Solution / Corrective Action:** Ensure motion is measured inside the configured region of interest around the deposition area. Check camera framing and region-of-interest configuration.
**Verification:** The motion score rises during actual deposition and printing status changes only after the configured consecutive-frame threshold.
**Important Parameters:** None numeric in the source.
**Safety / Important Notes:** Severity WARNING. Never tell the operator that a camera measurement is a confirmed machine state when the system is only inferring it from motion.
**Agent Follow-up:** Is the nozzle and deposition area inside the configured region of interest?
**English Questions:** The camera does not detect printing but the machine is running. No printing activity detected. Motion is not being seen.
**Hinglish Questions:** Machine chal rahi hai par camera printing detect nahi kar raha. Motion detect nahi ho raha.
**Source:** PDF 3, PRINT-020.
**Verification Status:** DOCUMENTED (project-derived). Cause is LIKELY, not operator-confirmed.

## Issue: Printing Pipe Needs Cleaning

**Problem:** Material flow becomes less reliable after extended operation, and a maintenance reminder appears.
**Symptoms / Observation:** Material flow becomes less reliable after extended operation; maintenance reminder appears.
**Cause:** The printing pipe has accumulated material or residue and requires routine cleaning.
**Solution / Corrective Action:** Clean the printing pipe every four hours of operation, as specified in the operating manual.
**Verification:** Flow is stable after cleaning and no material restriction is observed.
**Important Parameters:** Pipe cleaning interval: every 4 hours of operation.
**Safety / Important Notes:** Severity INFO. This is the PRINTING PIPE, cleaned every four HOURS. Do not confuse it with the ACCELERATOR HOLE, which is cleaned every five LAYERS with the big Allen key. They are different tasks with different intervals and different tools.
**Agent Follow-up:** When was the printing pipe last cleaned?
**English Questions:** How often should the printing pipe be cleaned? When was the pipe last cleaned? What is the pipe cleaning interval?
**Hinglish Questions:** Pipe ko last kab clean kiya tha? Printing pipe kitne ghante mein clean karna hai?
**Source:** PDF 3, PRINT-021.
**Verification Status:** DOCUMENTED (project-derived), consistent with the operating manual.

## Issue: Accelerator Coupling Maintenance

**Problem:** Routine accelerator coupling maintenance is due, or accelerator operation becomes questionable.
**Symptoms / Observation:** Routine maintenance is due or accelerator operation becomes questionable.
**Cause:** Coupling wear or looseness, or an overdue inspection.
**Solution / Corrective Action:** Inspect the accelerator coupling daily and correct any looseness or wear according to the machine maintenance procedure.
**Verification:** The coupling is secure and accelerator operation is stable.
**Important Parameters:** Accelerator coupling inspection interval: daily.
**Safety / Important Notes:** Severity INFO. This is the accelerator COUPLING, inspected DAILY. Do not confuse it with the accelerator HOLE, cleaned every five LAYERS.
**Agent Follow-up:** Has the accelerator coupling been checked today?
**English Questions:** How often should the accelerator coupling be inspected? Has the coupling been checked? Accelerator coupling maintenance.
**Hinglish Questions:** Accelerator coupling kab check karna hai? Coupling roz check karni hai?
**Source:** PDF 3, PRINT-022.
**Verification Status:** DOCUMENTED (project-derived), consistent with the operating manual.

## Issue: Pump Seal Maintenance

**Problem:** Persistent flow problems occur even after checking material and nozzle.
**Symptoms / Observation:** Persistent flow problems occur even after checking material and nozzle.
**Cause (LIKELY - NOT CONFIRMED):** The pump seal may be worn or failing.
**Solution / Corrective Action:** Inspect pump seals weekly and replace or service them according to the machine maintenance procedure if worn.
**Verification:** Stable flow is restored without repeated drops or pressure issues.
**Important Parameters:** Pump seal inspection interval: weekly.
**Safety / Important Notes:** Severity INFO. Distinct from cleaning the FRONT of the pump, which the site records specify after every two walls.
**Agent Follow-up:** When were the pump seals last inspected?
**English Questions:** How often should pump seals be inspected? When were the pump seals last checked? Pump seal maintenance.
**Hinglish Questions:** Pump seal kab check karna hai? Pump ki seal weekly check karni hai?
**Source:** PDF 3, PRINT-023.
**Verification Status:** DOCUMENTED (project-derived). Cause is LIKELY, not operator-confirmed.
