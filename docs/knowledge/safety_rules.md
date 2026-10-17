# Kelvin6K Safety Rules to Preserve

Sources: Kelvin6K_Printing_Data_Review_Problems_Solutions.pdf section 6, and
Kelvin6K_Printing_Issues_and_Solutions_Dataset(2).pdf section 5.

These rules take priority over any other answer. The assistant is monitoring and
advisory only and must never claim to have changed a machine setting.

## Issue: Machine and Print Safety Rules

**Problem:** Certain printing conditions are safety issues, not merely quality issues, and acting on them in the wrong order can injure someone or damage the machine.
**Symptoms / Observation:** A suspected nozzle clog, a critical machine error, severe delamination or layer shifting, or a machine-specific fault code.
**Cause:** Not applicable - these are standing rules.
**Solution / Corrective Action:** The documented safety rules:
  - Never clear a nozzle clog while the pump is pressurised. Stop the pump and release pressure first.
  - For critical machine errors, pause or stop the relevant process and inspect the machine before continuing. Do not continue blindly while a critical machine error is active.
  - Severe delamination or layer shifting should be physically inspected before additional layers are added.
  - Machine-specific commands should be treated as controlled procedures, not generic troubleshooting instructions. Identify the exact machine condition and follow the authorized procedure.
**Important Parameters:** Nozzle-clog procedure from the operating manual: stop the pump, release system pressure, soak the nozzle in solvent for ten minutes, and clear it with the cleaning rod.
**Safety / Important Notes:** AUTHORIZATION REQUIRED for machine-specific procedures, including the servo fault-clear parameter and the Z-height G-code corrections.
**English Questions:** What are the safety rules? Can I clear a clog while the pump is running? What do I do first in a machine error? Is it safe to keep printing after delamination?
**Hinglish Questions:** Safety rules kya hain? Pump chalu rehte hue clog saaf kar sakte hain? Machine error mein pehle kya karna hai?
**Source:** PDF 1 section 6; PDF 3 section 5.
**Verification Status:** DOCUMENTED SAFETY RULES.

## Issue: Temperature Safety Rule - Applicability Unconfirmed

**Problem:** The source documents include a safety rule about nozzle and bed temperature, but this machine may have neither.
**Symptoms / Observation:** None observed. No temperature-related incident appears in any of the 50 real site records.
**Cause:** Not applicable.
**Solution / Corrective Action:** The rule as written in the sources is: "High nozzle or bed temperature should be treated as a safety issue, not merely a print-quality issue." Confirm with the operator whether this machine has a heated nozzle or heated bed before acting on it.
**Important Parameters:** None specified in the sources.
**Safety / Important Notes:** APPLICABILITY UNCONFIRMED. Kelvin6K is a concrete printing machine with a pump, hopper and accelerator. Nozzle or bed temperature appears in none of the 50 real site incident records. The related entries PRINT-008 (high nozzle temperature), PRINT-009 (low nozzle temperature) and PRINT-010 (high bed temperature) were excluded from the knowledge base for that reason. This rule is retained only because it appears in the source documents; it must be confirmed with the operator before being acted on.
**English Questions:** Does this machine have a heated nozzle? Is nozzle temperature a safety issue? What is the maximum bed temperature?
**Hinglish Questions:** Is machine mein nozzle heater hai? Nozzle temperature ka rule hai?
**Source:** PDF 1 section 6; PDF 3 section 5 and PRINT-008 to PRINT-010.
**Verification Status:** APPLICABILITY UNCONFIRMED - pending operator confirmation. Likely not applicable to this machine.

## Issue: Site Safety and PPE

**Problem:** Site safety requirements were not consistently followed, and electrical hazards were left in place.
**Symptoms / Observation:** Labourers not wearing masks or gloves; a loose or open wire present for days; walls damaged during lifting; robot movement near a tent and wires.
**Cause:** Safety requirements not enforced.
**Solution / Corrective Action:** Enforce use of the provided PPE and site safety requirements. Install a proper plug or connector on any loose wire - no open wire should remain. Supervise lifting and crane movement and instruct labour on the correct lifting method. Move the tent before moving the machine, and take care of wires during movement.
**Important Parameters:** None specified in the source.
**Safety / Important Notes:** The source states safety equipment was provided. Open wiring is an electrical hazard and must not be left in place.
**English Questions:** What PPE is required on site? Is the open wire safe? How should walls be lifted safely?
**Hinglish Questions:** Site par kya safety equipment pehnana hai? Khula wire safe hai? Wall safely kaise uthayein?
**Source:** PDF 4 records 37, 38, 43, 45.
**Verification Status:** SITE-VERIFIED.
