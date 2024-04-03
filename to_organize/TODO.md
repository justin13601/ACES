TODO:
- computational profile
    create run_profiling so matthew can run on bigger dataset
	more variety, abnormal lab, weaning, progression
    total memory and cores allocated and machine details, chatgpt should be able to write to file
    use google spreadsheet -> csv so can plot using code
    
    check more variables:
        number of predicates/columns, number of patients/rows, types of configs, number of outputs rows (schema) -> structs and dataframe size	
	    number of inclusion/exclusion criteria (change last window)
            -> seems like number of criteria does not affect the query time (real nor fake criteria)
	    number of cores/resources (limit threads in polars -> polars max_threads (https://github.com/pola-rs/polars/issues/10847))
	make a script config + data dir + resource -> saves numbers as file

- validation
    check if the filtered out cohort is right
    check if final cohort is right
    convert to python script

- prompting
    provide more examples or generalize the current one
    -> Looks like providing multiple examples as context does help


Paper notes:
- realize an efficient way
- data expertise and clinical expertise to extract data 
- demonstrate efficacy
- future work, gpt, user eval; tractable to extract from natural language interface; Further research to appropriately communicate the elements in the data tables to design the appropriate predicates
- limitation of defining a true trigger window


ABNORMAL LABS
# Hemoglobin (Hb): 
- Normal range is 13.8 to 17.2 grams per deciliter (g/dL) for men and 12.1 to 15.1 g/dL for women. 
- Abnormal values indicate anemia (low) or polycythemia (high).

# White Blood Cell (WBC) count: 
- Normal range is 4,500 to 11,000 cells per microliter (µL). 
- Elevated counts suggest infection, while low counts indicate potential immunosuppression.

# Platelets: 
- Normal range is 150,000 to 450,000 platelets per µL. 
- Low platelet counts (thrombocytopenia) increase bleeding risk; high counts (thrombocytosis) may increase clotting risk.

# Sodium (Na+): 
- Normal range is 135 to 145 milliequivalents per liter (mEq/L). 
- Abnormalities can indicate fluid imbalance, renal dysfunction, or hormonal disturbances.

# Potassium (K+): 
- Normal range is 3.5 to 5.0 mEq/L. 
- Deviations can affect cardiac and muscular function.

# Chloride (Cl-): 
- Normal range is 98 to 106 mEq/L. 
- Changes may be associated with acid-base disturbances or dehydration.

# Creatinine: 
- Normal range is 0.9 to 1.3 mg/dL for men and 0.6 to 1.1 mg/dL for women.
- Elevated levels suggest renal impairment.

# Blood Urea Nitrogen (BUN): 
- Normal range is 10 to 20 mg/dL. 
- High values may indicate renal dysfunction or dehydration.

# Aspartate Aminotransferase (AST) and Alanine Aminotransferase (ALT): 
- Normal ranges are 10 to 40 units/L and 7 to 56 units/L. 
- Elevated levels indicate liver injury.

# Bilirubin: 
- Normal total bilirubin is 0.1 to 1.2 mg/dL. 
- High levels may indicate liver dysfunction or hemolysis.

# pH: 
- Normal arterial blood pH is 7.35 to 7.45.
- Values outside this range indicate acidosis (low) or alkalosis (high).

# Partial Pressure of Oxygen (PaO2): 
- Normal range is 75 to 100 mmHg. 
- Lower levels indicate hypoxemia.

# Partial Pressure of Carbon Dioxide (PaCO2): 
- Normal range is 35 to 45 mmHg. 
- Elevated or decreased levels can indicate respiratory dysfunction.

# Prothrombin Time (PT): 
- Normal range is 11 to 13.5 seconds. 
- Prolonged PT can indicate coagulation factor deficiency or anticoagulant therapy.

# Activated Partial Thromboplastin Time (aPTT): 
- Normal range is 30 to 40 seconds. 
- Like PT, prolonged values suggest coagulation factor deficiencies or the presence of inhibitors.

# Lactate: 
- Normal range is 0.5 to 2.2 mmol/L. 
- Elevated lactate levels are associated with tissue hypoxia and sepsis.