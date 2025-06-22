
import pandas as pd
import numpy as np
import sqlite3
import re
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

'''
Handles three prediction tasks: Mortality, ventilator, sepsis

Preprocessing steps:
1. Drug harmonization with reference drug list
2. Binary drug matrix creation
3. Demographics encoding (age categories, gender, ethnicity)
4. 48-hour time windows for mortality
5. 72-hour windows with 12-hour intervals for ventilator/sepsis  

'''

class eICUPreprocessor:
    """
    eICU preprocessor 
    Supports 3 tasks: mortality, ventilator, sepsis prediction
    """
    
    def __init__(self, demo_db_path: str, config: Dict):
        self.demo_db_path = demo_db_path
        self.config = config
        
        # top 10 hospitals with most patients
        self.target_hospitals = config.get('target_hospitals', 
            [167, 420, 199, 458, 252, 165, 148, 281, 449, 283])
        
        # common drugs reference 
        self.common_drugs = self._load_drug_reference()
        
        # lab tests (12 common between eICU and MIMIC-III)
        self.common_lab_tests = [
            'glucose', 'creatinine', 'bun', 'sodium', 'chloride', 'potassium',
            'hematocrit', 'hemoglobin', 'platelets', 'wbc', 'lactate', 'albumin'
        ]
        
        # time windows 
        self.time_windows = {
            'mortality': {'hours': 48, 'prediction_after': True},
            'ventilator': {'hours': 72, 'intervals': 6, 'interval_hours': 12},
            'sepsis': {'hours': 72, 'intervals': 6, 'interval_hours': 12}
        }
        
        self.prediction_task = config.get('prediction_task', 'mortality')
        
        if self.prediction_task not in ['mortality', 'ventilator', 'sepsis']:
            raise ValueError(f"Unsupported task: {self.prediction_task}. "
                           f"Supported: mortality, ventilator, sepsis")
    
    def _load_drug_reference(self) -> List[str]:
        '''
        Load  common drugs reference
        These are the drugs after harmonization in the original study
        '''
        # Core drugs (subset shown, expand to full 237)
        drugs = [
            # Cardiovascular drugs
            'aspirin', 'heparin', 'warfarin', 'enoxaparin', 'clopidogrel',
            'atorvastatin', 'metoprolol', 'lisinopril', 'amlodipine', 'furosemide',
            
            # Pain/Sedation
            'morphine', 'fentanyl', 'hydromorphone', 'acetaminophen', 'ibuprofen',
            'midazolam', 'lorazepam', 'propofol', 'dexmedetomidine',
            
            # Antibiotics (high importance for sepsis)
            'vancomycin', 'piperacillin', 'tazobactam', 'cefepime', 'meropenem',
            'ciprofloxacin', 'metronidazole', 'clindamycin', 'azithromycin',
            
            # ICU-specific drugs
            'norepinephrine', 'epinephrine', 'dopamine', 'vasopressin', 
            'phenylephrine', 'dobutamine',
            
            # Respiratory/Anesthesia (important for ventilator prediction)
            'vecuronium', 'cisatracurium', 'succinylcholine', 'rocuronium',
            'etomidate', 'ketamine', 'sevoflurane',
            
            # Diabetes/Endocrine
            'insulin', 'metformin', 'hydrocortisone', 'methylprednisolone',
            
            # Gastrointestinal
            'omeprazole', 'pantoprazole', 'ranitidine', 'ondansetron',
            
            # Others
            'chlorhexidine', 'glycopyrrolate', 'neostigmine', 'naloxone'
        ]
        
        # In production, load from actual FedWeight reference file
        return sorted(drugs)
    
    def load_eicu_tables(self) -> Dict[str, pd.DataFrame]:
        
        '''Load eICU demo tables from SQLite database'''
        
        print(f"Loading eICU demo data from {self.demo_db_path}")
        
        try:
            conn = sqlite3.connect(self.demo_db_path)
            
            tables = {
                'patient': pd.read_sql_query("SELECT * FROM patient", conn),
                'medication': pd.read_sql_query("SELECT * FROM medication", conn),
                'diagnosis': pd.read_sql_query("SELECT * FROM diagnosis", conn),
                'treatment': pd.read_sql_query("SELECT * FROM treatment", conn)
            }
            
            try:
                tables['lab'] = pd.read_sql_query("SELECT * FROM lab", conn)
            except:
                print("Warning: Lab table not found in demo data")
                tables['lab'] = pd.DataFrame()
            
            conn.close()
            
            for table_name, df in tables.items():
                print(f"  {table_name}: {len(df)} rows, {df.shape[1]} columns")
            
            return tables
            
        except Exception as e:
            raise RuntimeError(f"Failed to load eICU data: {e}")
        
    
    def harmonize_drug_names(self, medication_df: pd.DataFrame) -> pd.DataFrame:
        ''' Should use BioVec Embedding when mapping, but for simplicity, we just use the reference list '''
        print("Applying drug harmonization...")
        
        def harmonize_drug(drug_name):
            if pd.isna(drug_name) or drug_name == '':
                return 'unknown'
            
            drug_clean = str(drug_name).lower().strip()
            
            # Remove dosage information 
            drug_clean = re.sub(r'\d+\s*(mg|mcg|g|ml|units?|iu|meq)\b', '', drug_clean)
            drug_clean = re.sub(r'\d+\.\d+\s*(mg|mcg|g|ml|units?|iu|meq)\b', '', drug_clean)
            drug_clean = re.sub(r'\d+', '', drug_clean).strip()
            
            # Remove common suffixes/prefixes
            drug_clean = re.sub(r'\b(iv|po|oral|injection|tablet|capsule)\b', '', drug_clean)
            drug_clean = re.sub(r'\s+', ' ', drug_clean).strip()
            
            if not drug_clean or drug_clean.isspace():
                return 'unknown'
            
            # Direct mapping to reference (step 2)
            for ref_drug in self.common_drugs:
                if ref_drug in drug_clean or drug_clean in ref_drug:
                    return ref_drug
            
            # Handle compound drugs (e.g., piperacillin-tazobactam)
            if '-' in drug_clean:
                parts = drug_clean.split('-')
                for part in parts:
                    part = part.strip()
                    for ref_drug in self.common_drugs:
                        if ref_drug in part or part in ref_drug:
                            return ref_drug
            
            # Similarity matching for remaining drugs (Use BioVec Embedding when available!!)
            best_match, best_score = None, 0
            for ref_drug in self.common_drugs:
                words1 = set(drug_clean.split())
                words2 = set(ref_drug.split())
                if words1 and words2:
                    similarity = len(words1 & words2) / len(words1 | words2)
                    if similarity > best_score and similarity > 0.4:
                        best_score = similarity
                        best_match = ref_drug
            
            return best_match if best_match else 'other'
        
        # Apply harmonization
        medication_df = medication_df.copy()
        medication_df['drugname_harmonized'] = medication_df['drugname'].apply(harmonize_drug)
        
        # Print harmonization statistics
        original_drugs = len(medication_df['drugname'].unique())
        harmonized_drugs = len(medication_df['drugname_harmonized'].unique())
        print(f"  Drug harmonization: {original_drugs} -> {harmonized_drugs} unique drugs")
        
        # Show distribution of harmonized drugs
        drug_counts = medication_df['drugname_harmonized'].value_counts()
        print(f"  Top 10 drugs after harmonization:")
        for drug, count in drug_counts.head(10).items():
            print(f"    {drug}: {count}")
        
        return medication_df
    
    def encode_patient_demographics(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode demographics:
        
        - Age: 8 categories (<30, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, >89)
        - Gender: Binary (Male=1, Female=0)  
        - Ethnicity: 5 categories (Caucasian, African American, Hispanic, Asian, Native American)
        """
        print("Encoding demographics...")
        
        patient_processed = patient_df.copy()
        
        # Age categories 
        def categorize_age(age):
            if pd.isna(age) or age == '' or age == 'NULL':
                return -1  # Unknown
            try:
                age_val = float(age)
                if age_val < 30: return 0      # <30
                elif age_val < 40: return 1    # 30-39
                elif age_val < 50: return 2    # 40-49
                elif age_val < 60: return 3    # 50-59
                elif age_val < 70: return 4    # 60-69
                elif age_val < 80: return 5    # 70-79
                elif age_val < 90: return 6    # 80-89
                else: return 7                 # >89
            except (ValueError, TypeError):
                return -1
        
        patient_processed['age_category'] = patient_df['age'].apply(categorize_age)
        
        # Gender binary 
        patient_processed['gender_binary'] = patient_df['gender'].map({
            'Male': 1, 'Female': 0, 'M': 1, 'F': 0
        }).fillna(-1)
        
        # Ethnicity encoding 
        ethnicity_map = {
            'Caucasian': 0,
            'African American': 1, 
            'Hispanic': 2,
            'Asian': 3,
            'Native American': 4,
            'Other': 5,
            'Unknown': 5
        }
        
        # Handle variations in ethnicity naming
        def map_ethnicity(eth):
            if pd.isna(eth) or eth == '' or eth == 'NULL':
                return 5
            eth_str = str(eth).strip()
            # Direct mapping
            if eth_str in ethnicity_map:
                return ethnicity_map[eth_str]
            # Fuzzy matching for variations
            eth_lower = eth_str.lower()
            if 'caucasian' in eth_lower or 'white' in eth_lower:
                return 0
            elif 'african' in eth_lower or 'black' in eth_lower:
                return 1
            elif 'hispanic' in eth_lower or 'latino' in eth_lower:
                return 2
            elif 'asian' in eth_lower:
                return 3
            elif 'native' in eth_lower or 'american indian' in eth_lower:
                return 4
            else:
                return 5
        
        patient_processed['ethnicity_encoded'] = patient_df['ethnicity'].apply(map_ethnicity)
        
        # Print encoding statistics
        print(f"  Age categories distribution:")
        age_dist = patient_processed['age_category'].value_counts().sort_index()
        age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '>89', 'Unknown']
        for idx, count in age_dist.items():
            label = age_labels[idx] if idx >= 0 else 'Unknown'
            print(f"    {label}: {count}")
        
        print(f"  Gender distribution:")
        gender_dist = patient_processed['gender_binary'].value_counts()
        for gender, count in gender_dist.items():
            label = 'Male' if gender == 1 else 'Female' if gender == 0 else 'Unknown'
            print(f"    {label}: {count}")
        
        return patient_processed
    
    
    def create_time_windows(self, 
                                    patient_df: pd.DataFrame,
                                    medication_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create time windows
        
         time windows:
        - Mortality: 48-hour observation window, predict after 48h
        - Ventilator: 72-hour observation, 6 intervals of 12h each  
        - Sepsis: 72-hour observation, 6 intervals of 12h each
        """
        print(f"Creating time windows for {self.prediction_task} prediction...")
        
        window_config = self.time_windows[self.prediction_task]
        
        if self.prediction_task == 'mortality':
            return self._create_mortality_windows(patient_df, medication_df, window_config)
        elif self.prediction_task in ['ventilator', 'sepsis']:
            return self._create_interval_windows(patient_df, medication_df, window_config)
    
    def _create_mortality_windows(self, 
                                patient_df: pd.DataFrame,
                                medication_df: pd.DataFrame,
                                config: Dict) -> Dict[str, pd.DataFrame]:
        """
        Create 48-hour windows for mortality prediction 
        """
        window_hours = config['hours']
        window_minutes = window_hours * 60
        
        print(f"  Creating {window_hours}-hour windows for mortality prediction")
        
        # Filter medications to first 48 hours
        if 'drugstartoffset' in medication_df.columns:
            medication_windowed = medication_df[
                medication_df['drugstartoffset'] <= window_minutes
            ].copy()
        else:
            # If no time offset, use all medications
            medication_windowed = medication_df.copy()
            print("  Warning: No drugstartoffset found, using all medications")
        
        print(f"  Medications in {window_hours}h window: {len(medication_windowed)}")
        
        return {
            'patient': patient_df,
            'medication': medication_windowed
        }
    
    def _create_interval_windows(self,
                               patient_df: pd.DataFrame, 
                               medication_df: pd.DataFrame,
                               config: Dict) -> Dict[str, pd.DataFrame]:
        """
        Create 72-hour windows with 12-hour intervals for ventilator/sepsis prediction
        """
        total_hours = config['hours']
        num_intervals = config['intervals'] 
        interval_hours = config['interval_hours']
        
        print(f"  Creating {total_hours}h observation with {num_intervals} intervals of {interval_hours}h")
        
        # For demo, we'll use the first 72 hours of data
        # In practice, this would create multiple windows per patient
        total_minutes = total_hours * 60
        
        if 'drugstartoffset' in medication_df.columns:
            medication_windowed = medication_df[
                medication_df['drugstartoffset'] <= total_minutes
            ].copy()
            
            # Add interval information for future use
            medication_windowed['interval'] = (
                medication_windowed['drugstartoffset'] // (interval_hours * 60)
            ).astype(int)
        else:
            medication_windowed = medication_df.copy()
            medication_windowed['interval'] = 0  # Default interval
            print("  Warning: No drugstartoffset found, assigning to interval 0")
        
        print(f"  Medications in {total_hours}h window: {len(medication_windowed)}")
        
        return {
            'patient': patient_df,
            'medication': medication_windowed
        }
        
    def create_binary_drug_matrix(self,
                                          patient_df: pd.DataFrame,
                                          medication_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create binary drug matrix
        
        Returns:
            Binary matrix (patients x drugs) and drug names list
        """
        print("Creating binary drug matrix...")
        
        unique_drugs = sorted(medication_df['drugname_harmonized'].unique())
        print(f"  Unique drugs after harmonization: {len(unique_drugs)}")
        
        # Create binary matrix
        drug_matrix = []
        valid_patients = []
        
        for _, patient in patient_df.iterrows():
            patient_id = patient['patientunitstayid']
            
            # patient medications
            patient_meds = medication_df[
                medication_df['patientunitstayid'] == patient_id
            ]
            
            drug_vector = []
            for drug in unique_drugs:
                has_drug = 1 if drug in patient_meds['drugname_harmonized'].values else 0
                drug_vector.append(has_drug)
            
            drug_matrix.append(drug_vector)
            valid_patients.append(patient_id)
        
        drug_matrix = np.array(drug_matrix, dtype=np.float32)
        
        print(f"  Binary drug matrix shape: {drug_matrix.shape}")
        print(f"  Sparsity: {(drug_matrix == 0).mean():.3f}")
        
        return drug_matrix, unique_drugs
    
    def create_demographic_features(self, patient_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create demographic features
        """
        demo_features = []
        feature_names = []
        
        for _, patient in patient_df.iterrows():
            patient_features = []
            
            # Age categories (one-hot encoding)
            age_cat = patient.get('age_category', -1)
            for i in range(8):  # 8 age categories
                patient_features.append(1.0 if age_cat == i else 0.0)
            
            patient_features.append(float(patient.get('gender_binary', -1)))
            
            # Ethnicity  
            patient_features.append(float(patient.get('ethnicity_encoded', 5)))
            
            demo_features.append(patient_features)
        
        # Feature names
        age_names = [f"age_{i}" for i in range(8)]
        feature_names = age_names + ['gender', 'ethnicity']
        
        return np.array(demo_features, dtype=np.float32), feature_names
    
    def create_labels(self, patient_df: pd.DataFrame) -> np.ndarray:
        """
        Create labels for the specified prediction task
        """
        labels = []
        
        for _, patient in patient_df.iterrows():
            if self.prediction_task == 'mortality':
                # Hospital discharge status for mortality
                status = patient.get('hospitaldischargestatus', 'Unknown')
                label = 1 if status == 'Expired' else 0
                
            elif self.prediction_task == 'ventilator':
                # For demo, we'll simulate ventilator labels
                # In practice, derive from treatment table
                label = 0  # Default, would need treatment data
                
            elif self.prediction_task == 'sepsis':
                # For demo, we'll simulate sepsis labels  
                # In practice, derive from diagnosis table
                label = 0  # Default, would need diagnosis data
                
            labels.append(label)
        
        return np.array(labels, dtype=np.int64)
    
    def partition_by_hospitals(self,
                                patient_df: pd.DataFrame,
                                medication_df: pd.DataFrame) -> Dict[int, Dict]:
        """
        Partition data by hospitals
        """
        print("Partitioning data by hospitals...")
        
        hospital_datasets = {}
        
        for hospital_id in self.target_hospitals:
            print(f"  Processing hospital {hospital_id}...")
            
            # Get hospital patients
            if 'hospitalid' in patient_df.columns:
                hospital_patients = patient_df[patient_df['hospitalid'] == hospital_id]
            else:
                # Simulate hospital assignment
                start_id = hospital_id * 50  # Adjust range as needed
                end_id = (hospital_id + 1) * 50
                hospital_patients = patient_df[
                    (patient_df['patientunitstayid'] >= start_id) & 
                    (patient_df['patientunitstayid'] < end_id)
                ]
            
            if len(hospital_patients) < 5:  # Minimum patients threshold
                print(f"    Skipping hospital {hospital_id}: only {len(hospital_patients)} patients")
                continue
            
            # Get hospital medications
            patient_ids = hospital_patients['patientunitstayid'].values
            hospital_meds = medication_df[
                medication_df['patientunitstayid'].isin(patient_ids)
            ]
            
            # Create features and labels for this hospital
            drug_matrix, drug_names = self.create_binary_drug_matrix(
                hospital_patients, hospital_meds
            )
            demo_features, demo_names = self.create_demographic_features(
                hospital_patients
            )
            labels = self.create_labels(hospital_patients)
            
            # Combine features
            X = np.concatenate([drug_matrix, demo_features], axis=1)
            feature_names = drug_names + demo_names
            
            # Remove patients with no features or invalid labels
            valid_mask = (X.sum(axis=1) > 0) & (labels >= 0)
            X_valid = X[valid_mask]
            y_valid = labels[valid_mask]
            
            if len(X_valid) == 0:
                print(f"    Skipping hospital {hospital_id}: no valid patients")
                continue
            
            hospital_datasets[hospital_id] = {
                'X': X_valid,
                'y': y_valid,
                'feature_names': feature_names,
                'num_samples': len(X_valid),
                'num_features': X_valid.shape[1],
                'num_drug_features': len(drug_names),
                'num_demo_features': len(demo_names)
            }
            
            print(f"    Hospital {hospital_id}: {len(X_valid)} patients, "
                  f"{X_valid.shape[1]} features, {len(np.unique(y_valid))} classes")
        
        print(f"Successfully processed {len(hospital_datasets)} hospitals")
        return hospital_datasets
    
    
    def preprocess_eicu(self) -> Dict:
        """
        Main preprocessing function
        
        Returns:
            Preprocessed data ready for FedFed training
        """
        print(f"\n=== eICU Preprocessing for {self.prediction_task.upper()} prediction ===")
        
        # Step 1: Load raw data
        tables = self.load_eicu_tables()
        
        # Step 2: Harmonize drug names 
        medication_df = self.harmonize_drug_names(tables['medication'])
        
        # Step 3: Encode demographics
        patient_df = self.encode_patient_demographics(tables['patient'])
        
        # Step 4: Create time windows 
        windowed_data = self.create_time_windows(patient_df, medication_df)
        
        # Step 5: Partition by hospitals 
        hospital_datasets = self.partition_by_hospitals(
            windowed_data['patient'], windowed_data['medication']
        )
        
        # Prepare final output
        result = {
            'hospital_datasets': hospital_datasets,
            'metadata': {
                'prediction_task': self.prediction_task,
                'preprocessing_method': 'following fedweight',
                'time_window_config': self.time_windows[self.prediction_task],
                'num_hospitals': len(hospital_datasets),
                'total_patients': sum(h['num_samples'] for h in hospital_datasets.values()),
                'feature_categories': {
                    'drugs': len(self.common_drugs),
                    'demographics': 10  # 8 age + gender + ethnicity
                }
            }
        }
        
        print(f"\n=== Preprocessing Complete ===")
        print(f"Task: {self.prediction_task}")
        print(f"Hospitals: {len(hospital_datasets)}")
        print(f"Total patients: {result['metadata']['total_patients']}")
        print(f"Features per patient: {list(hospital_datasets.values())[0]['num_features']}")
        
        return result


# ----------------------- Main Function -----------------------
def preprocess_eicu(demo_db_path: str, config: Dict) -> Dict:
    """    
    Args:
        demo_db_path: Path to eICU demo database
        config: Configuration dictionary with task and parameters
        
    Returns:
        Preprocessed data ready for FedFed
    """
    preprocessor = eICUPreprocessor(demo_db_path, config)
    return preprocessor.preprocess_eicu()