import pandas as pd
import fasttext
import numpy as np
from tqdm import tqdm
import re

unique_drugs = ['acetamin', 'biotene', 'compazine', 'ferrous', 'imdur', 'lidocaine', 'milk of magnesia', 'nystatin', 'prochlorperazine', 'tamsulosin',
                'advair diskus', 'bisacodyl', 'coreg', 'flagyl', 'influenza vac', 'lipitor', 'mineral', 'omeprazole', 'promethazine', 'thiamine',
                'albumin', 'bumetanide', 'cozaar', 'flomax', 'infuvite', 'lisinopril', 'mineral oil', 'ondansetron', 'propofol', 'ticagrelor',
                'albuterol', 'bumex', 'decadron', 'flumazenil', 'insulin', 'lispro', 'mono-sod', 'optiray', 'pulmicort respule', 'tiotropium',
                'allopurinol', 'buminate', 'definity', 'fluticasone-salmeterol', 'insulin detemir', 'loratadine', 'morphine', 'oxycodone', 'quetiapine', 'toradol',
                'alprazolam', 'calcium carbonate', 'deltasone', 'folic acid', 'iohexol', 'lorazepam', 'motrin', 'pantoprazole', 'refresh p.m. op oint', 'tramadol',
                'alteplase', 'calcium chloride', 'dexamethasone', 'furosemide', 'iopamidol', 'losartan', 'mupirocin', 'parenteral nutrition', 'reglan', 'trandate',
                'alum hydroxide', 'calcium gluconate', 'dexmedetomidine', 'gabapentin', 'ipratropium', 'maalox', 'nafcillin', 'percocet', 'restoril', 'transde rm-scop',
                'ambien', 'cardizem', 'dextrose', 'glargine', 'isosorbide', 'magnesium chloride', 'naloxone', 'phenergan', 'ringers solution', 'trazodone',
                'aminocaproic acid', 'carvedilol', 'diazepam', 'glucagen', 'kayciel', 'magnesium hydroxide', 'narcan', 'phenylephrine', 'rocuronium', 'ultram',
                'amiodarone', 'catapres', 'digoxin', 'glucagon', 'kayexalate', 'magnesium oxide', 'neostigmine', 'phytonadione', 'roxicodone', 'valium',
                'amlodipine', 'cefazolin', 'diltiazem', 'glucose', 'keppra', 'magnesium sulf', 'neostigmine methylsulfate', 'piperacillin', 'sennosides', 'vancomycin',
                'anticoagulant', 'cefepime', 'diphenhydramine', 'glycopyrrolate', 'ketorolac', 'magox', 'neurontin', 'plasmalyte', 'seroquel', 'vasopressin',
                'apresoline', 'ceftriaxone', 'diprivan', 'guaifenesin', 'klonopin', 'medrol', 'nexterone', 'plavix', 'sertraline', 'ventolin',
                'ascorbic acid', 'cephulac', 'docusate', 'haldol', 'labetalol', 'meperidine', 'nicardipine', 'pneumococcal', 'simethicone', 'vitamin',
                'aspart', 'cetirizine', 'dopamine', 'haloperidol', 'lactated ringer', 'meropenem', 'nicoderm', 'pnu-immune-23', 'simvastatin', 'warfarin',
                'aspirin', 'chlorhexidine', 'ecotrin', 'heparin', 'lactulose', 'merrem', 'nicotine', 'polyethylene glycol', 'sodium bicarbonate', 'xanax',
                'atenolol', 'ciprofloxacin', 'enoxaparin', 'humulin', 'lanoxin', 'metformin', 'nitro-bid', 'potassium chloride', 'sodium chloride', 'zestril',
                'atorvastatin', 'cisatracurium', 'ephedrine', 'hydralazine', 'lantus', 'methylprednisolone', 'nitroglycerin', 'potassium phosphate', 'sodium phosphate', 'zocor',
                'atropine', 'citalopram', 'epinephrine', 'hydrochlorothiazide', 'levaquin', 'metoclopramide', 'nitroprusside', 'pravastatin', 'polystyrene sulfonate', 'zolpidem',
                'atrovent', 'clindamycin', 'etomidate', 'hydrocodone', 'levemir', 'metoprolol', 'norco', 'precedex', 'spironolactone', 'zosyn',
                'azithromycin', 'clonazepam', 'famotidine', 'hydrocortisone', 'levetiracetam', 'metronidazole', 'norepinephrine', 'prednisone', 'sublimaze',
                'bacitracin', 'clonidine', 'fat emulsion', 'hydromorphone', 'levofloxacin', 'midazolam', 'normodyne', 'prilocaine', 'succinylcholine',
                'bayer chewable', 'clopidogrel', 'fentanyl', 'ibuprofen', 'levothyroxine', 'midodrine', 'norvasc', 'prinivil', 'tacrolimus']


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def calculate_similarity(model, drug1, drug2):
    drug1_vector = model.get_word_vector(drug1)
    drug2_vector = model.get_word_vector(drug2)
    return cosine_similarity(drug1_vector, drug2_vector)


def get_age_group(age):

    if pd.isna(age):
        # 60 - 69 has the most patients
        return "60 - 69"

    if age == "> 89":
        return age

    age = int(age)

    if age < 30:
        return "< 30"
    elif age < 40:
        return "30 - 39"
    elif age < 50:
        return "40 - 49"
    elif age < 60:
        return "50 - 59"
    elif age < 70:
        return "60 - 69"
    elif age < 80:
        return "70 - 79"
    elif age < 90:
        return "80 - 89"
    else:
        return "> 89"


def get_bmi(weight_kg, height_cm):

    if pd.isna(weight_kg) or pd.isna(height_cm) or weight_kg == 0 or height_cm == 0:
        return "bmi_normal"

    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        return "bmi_underweight"
    elif bmi < 24.9:
        return "bmi_normal"
    elif bmi < 29.9:
        return "bmi_overweight"
    else:
        return "bmi_obesity"


def get_race(race):

    if pd.isna(race) or race == "Caucasian":
        return "race_caucasion"
    elif race == "African American":
        return "race_african"
    elif race == "Hispanic":
        return "race_hispanic"
    elif race == "Asian":
        return "race_asian"
    elif race == "Native American":
        return "race_native"
    else:
        return "race_caucasion"


def harmonize_drug(drug, unique_drugs, pretrained):

    if pd.isna(drug):
        return None

    drug = drug.lower()

    # Direct mapping: check whether drug names contain unique_drug
    for unique_drug in unique_drugs:
        if unique_drug in drug:
            return unique_drug

    # Check cosine similarity between word embeddings
    converted = re.sub(r'[^a-zA-Z\s]', '', drug)
    best_similarity = float('-inf')
    best_drug = None
    for drug_name in converted.split():
        for unique_drug in unique_drugs:
            similarity = calculate_similarity(
                pretrained, drug_name, unique_drug)
            if similarity > best_similarity:
                best_similarity = similarity
                best_drug = unique_drug

    return best_drug


def main():

    patient = pd.read_csv("data/eicu/patient.csv.gz")
    medication = pd.read_csv("data/eicu/medication.csv.gz")

    # Filter treatment after 48 hours
    treatment = pd.read_csv("data/eicu/treatment.csv.gz")
    treatment = treatment[treatment['treatmentoffset'] >= 48 * 60]
    ventilator = treatment[treatment['treatmentstring'].str.contains('ventilation')]
    ventilator_patients = ventilator['patientunitstayid'].unique()

    # Filter diagnosis after 48 hours
    diagnosis = pd.read_csv("data/eicu/diagnosis.csv.gz")
    diagnosis = diagnosis[diagnosis['diagnosisoffset'] >= 48 * 60]
    sepsis = diagnosis[diagnosis['diagnosisstring'].str.contains('sepsis')]
    sepsis_patients = sepsis['patientunitstayid'].unique()

    # Impute missing drugs
    print("Impute missing drugs...")
    available_drug_hicl = medication[~medication["drugname"].isna() & ~medication["drughiclseqno"].isna()]
    available_drug_hicl_dict = available_drug_hicl.set_index('drughiclseqno')['drugname'].to_dict()

    medication_imputed = medication.copy()
    mask = medication['drugname'].isna() & medication['drughiclseqno'].notna()
    medication_imputed.loc[mask, 'drugname'] = medication_imputed.loc[mask, 'drughiclseqno'].map(available_drug_hicl_dict)
    medication_imputed.to_csv('output/medication_imputed.csv', index=False)

    # Data harmonize
    pretrained = fasttext.load_model("data/pretrained/BioWordVec_PubMed_MIMICIII_d200.bin")

    print("Create eicu dataset...")
    eicu = pd.DataFrame(columns=['patientunitstayid', 'hospitalid', 'death', 'ventilation', 'sepsis'] + unique_drugs + ['bmi_underweight', 'bmi_normal', 'bmi_overweight', 'bmi_obesity', 'race_african',
                        'race_hispanic', 'race_caucasion', 'race_asian', 'race_native', 'sex_is_male', 'sex_is_female', '< 30', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70 - 79', '80 - 89', '> 89'])

    patient_medication = patient.merge(medication_imputed, on='patientunitstayid', how='inner')
    patient_medication_groups = patient_medication.groupby('patientunitstayid')["drugname"].apply(list)

    for index, row in tqdm(patient.iterrows(), total=patient.shape[0]):

        patientunitstayid = row['patientunitstayid']

        eicu_row = pd.DataFrame(0.0, index=[0], columns=eicu.columns)

        eicu_row['patientunitstayid'] = float(patientunitstayid)
        eicu_row['hospitalid'] = float(row['hospitalid'])

        # Labels
        eicu_row['death'] = float(row['unitdischargestatus'] == 'Expired')
        eicu_row['ventilation'] = float(patientunitstayid in ventilator_patients)
        eicu_row['sepsis'] = float(patientunitstayid in sepsis_patients)

        # BMI
        bmi = get_bmi(row['admissionweight'], row['admissionheight'])
        eicu_row[bmi] = 1.0

        # Race
        race = get_race(row['ethnicity'])
        eicu_row[race] = 1.0

        # Sex
        sex = "sex_is_female" if row['gender'] == 'Female' else "sex_is_male"
        eicu_row[sex] = 1.0

        # Age
        age_group = get_age_group(row['age'])
        eicu_row[age_group] = 1.0

        # Drugs
        drugs = patient_medication_groups.get(patientunitstayid)
        if drugs is not None and len(drugs) > 0:
            for drug in drugs:
                converted_drug = harmonize_drug(drug, unique_drugs, pretrained)
                if converted_drug is not None:
                    eicu_row[converted_drug] = 1.0

        eicu = pd.concat([eicu, eicu_row], ignore_index=True)

    print("Export harmonized eicu dataset...")
    eicu.to_csv('output/eicu_harmonized.csv.gz', index=False)


if __name__ == "__main__":
    main()