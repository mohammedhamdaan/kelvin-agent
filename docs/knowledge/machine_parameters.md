# Kelvin6K Observed Machine Parameters and Timing

Source: 3D_Concrete_Printing_Agent_Training_Dataset(2).pdf, sections 1 and 2.

IMPORTANT: these are values READ OFF THE MACHINE SCREEN during one observation
of one print job. They are a record of what the machine UI displays, not
production standards. Kelvin6K_Printing_Data_Review_Problems_Solutions.pdf marks
the 600 total layers as CONTEXT-SPECIFIC and the 1 minute 30 second layer time
as OBSERVATION ONLY.

These are NOT live values. The assistant must never answer a question about the
CURRENT machine state from this file.

## Issue: Machine Parameter Snapshot from the Operator Screen

**Problem:** Reference record of which parameters the machine UI displays, and the values observed during one print.
**Symptoms / Observation:** Observed during printing on the machine operator screen. Machine status Running, Mode AUTO.
**Cause:** Not applicable - this is a reference record, not a fault.
**Solution / Corrective Action:** Not applicable. Use this only to know which fields the machine reports. For the current values, read live machine telemetry, which is not yet connected to Kelvin6K.
**Important Parameters:** Machine status Running. Mode AUTO. Extruder 70 percent. Pump 24 percent. Fluid Pump 0 percent. Feedrate Override 150 percent. Current Feedrate 3600. Max Velocity 6000. Max Rapid 6000. Progress 20 percent. Completed Layers 115. Remaining Layers 485. Total layers indicated 600. Machine Run Time shown 00:10:42. Estimated Time shown 221.4 minutes. X position 1868.600. Y position 655.041. Z position 300.000. Velocity shown 3600.
**Safety / Important Notes:** CONTEXT-SPECIFIC VALUES. The 600 total layers belongs to that specific print job and must not be treated as a universal wall layer count. The 150 percent feedrate override here is a snapshot reading, which is a different thing from the 150 percent recorded after a speed correction in the site incidents; see the conflicts file. None of these values describe the machine right now.
**English Questions:** What parameters does the machine display? What was the extruder percentage? How many layers were completed? What does the machine UI show? What is the max velocity?
**Hinglish Questions:** Machine screen par kya kya dikhta hai? Extruder kitne percent tha? Kitni layers complete hui thi?
**Source:** PDF 2, Section 2.
**Verification Status:** OBSERVED READING from one print, one moment. NOT a production standard. NOT live data.

## Issue: Observed Layer Printing Time

**Problem:** Reference record of how long one observed layer took to print.
**Symptoms / Observation:** A 5-minute timer was used while observing one layer. At the end of the observation 3 minutes 30 seconds remained on the timer, so the elapsed printing time was 5:00 minus 3:30, which is 1:30.
**Cause:** Not applicable - this is an observation, not a fault.
**Solution / Corrective Action:** Not applicable. Do not use this as a standard layer time or as the basis of an estimate without confirmation.
**Important Parameters:** Timer duration 5 minutes. Time remaining 3 minutes 30 seconds. Elapsed time 1 minute 30 seconds. Observed layer time approximately 1 minute 30 seconds per layer.
**Safety / Important Notes:** OBSERVATION ONLY. This is a single observation of one layer and is not yet a standard layer time. The data review document marks it explicitly as observation only.
**English Questions:** How long does one layer take? What is the layer time? How many minutes per layer?
**Hinglish Questions:** Ek layer kitni der mein print hoti hai? Layer time kitna hai? Ek layer mein kitna time lagta hai?
**Source:** PDF 2, Section 1.
**Verification Status:** SINGLE OBSERVATION. NOT a confirmed standard.
