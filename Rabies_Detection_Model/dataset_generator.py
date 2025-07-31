import pandas as pd
import numpy as np
import random

# --- Configuration ---
NUM_RECORDS = 3000
RABIES_PREVALENCE = 0.15  # Target prevalence for rabies in the dataset
OUTPUT_FILENAME = 'rabies_risk_dataset_realistic_symptoms_3000.csv'

# --- Feature Lists ---
dog_breeds = [
    'Labrador Retriever', 'German Shepherd', 'Golden Retriever', 'French Bulldog',
    'Bulldog', 'Poodle', 'Beagle', 'Rottweiler', 'Dachshund', 'Boxer',
    'Siberian Husky', 'Chihuahua', 'Shih Tzu', 'Great Dane', 'Pomeranian',
    'Australian Shepherd', 'Doberman Pinscher', 'Mixed Breed', 'Stray'
]

# --- Helper Functions ---

def generate_base_data(num_records):
    """Generates the foundational data like breed, age, and initial risk factors."""
    data = []
    for _ in range(num_records):
        breed = random.choice(dog_breeds)
        age = random.randint(1, 15)
        
        # Strays are less likely to be groomed or vaccinated
        if breed == 'Stray':
            vaccinated = np.random.choice(['no', 'yes'], p=[0.85, 0.15])
            grooming = np.random.choice(['no', 'yes'], p=[0.9, 0.1])
        else:
            vaccinated = np.random.choice(['yes', 'no'], p=[0.85, 0.15])
            grooming = np.random.choice(['yes', 'no'], p=[0.7, 0.3])
            
        data.append([breed, age, vaccinated, grooming])
        
    df = pd.DataFrame(data, columns=['dog_breed', 'dog_age', 'vaccinated', 'grooming'])
    return df

def assign_rabies(df):
    """Assigns the target variable 'rabies' based on risk factors."""
    rabies_status = []
    
    # Calculate probabilities for each dog
    for _, row in df.iterrows():
        prob = 0.01 # Base probability
        if row['vaccinated'] == 'no':
            prob += 0.40 # Major risk factor
        if row['dog_breed'] == 'Stray':
            prob += 0.25 # Additional risk for strays
        
        # Ensure probability is capped at a reasonable level
        prob = min(prob, 0.8)

        rabies_status.append(np.random.choice(['yes', 'no'], p=[prob, 1-prob]))
        
    df['rabies'] = rabies_status
    
    # Adjust to meet target prevalence if necessary
    current_prevalence = df['rabies'].value_counts(normalize=True).get('yes', 0)
    num_rabid = int(len(df) * RABIES_PREVALENCE)
    
    if len(df[df['rabies'] == 'yes']) > num_rabid:
        # Too many rabid dogs, flip some 'yes' to 'no' (preferably vaccinated ones)
        indices_to_flip = df[(df['rabies'] == 'yes') & (df['vaccinated'] == 'yes')].index
        num_to_flip = len(df[df['rabies'] == 'yes']) - num_rabid
        if len(indices_to_flip) < num_to_flip:
             indices_to_flip = df[df['rabies'] == 'yes'].index
        
        flip_indices = np.random.choice(indices_to_flip, size=min(num_to_flip, len(indices_to_flip)), replace=False)
        df.loc[flip_indices, 'rabies'] = 'no'
        
    elif len(df[df['rabies'] == 'yes']) < num_rabid:
        # Too few rabid dogs, flip some 'no' to 'yes' (preferably unvaccinated ones)
        indices_to_flip = df[(df['rabies'] == 'no') & (df['vaccinated'] == 'no')].index
        num_to_flip = num_rabid - len(df[df['rabies'] == 'yes'])
        if len(indices_to_flip) < num_to_flip:
             indices_to_flip = df[df['rabies'] == 'no'].index
             
        flip_indices = np.random.choice(indices_to_flip, size=min(num_to_flip, len(indices_to_flip)), replace=False)
        df.loc[flip_indices, 'rabies'] = 'yes'

    return df

def generate_symptoms(df):
    """Generates realistic, staged symptom data based on whether the dog has rabies."""
    # Define symptoms by stage of rabies progression
    early_symptoms = ['fever', 'lethargy', 'behavior_change']
    furious_symptoms = ['aggression', 'excessive_drooling', 'difficulty_swallowing', 'seizures']
    paralytic_symptoms = ['staggering_gait', 'hind_leg_paralysis', 'jaw_dropped']
    
    all_symptoms = early_symptoms + furious_symptoms + paralytic_symptoms
    
    for symptom in all_symptoms:
        df[symptom] = 'no'

    rabid_indices = df[df['rabies'] == 'yes'].index

    for index in rabid_indices:
        # Assign a stage of rabies to each rabid dog
        stage = random.choice(['early', 'furious', 'paralytic'])
        
        symptoms_to_show = []
        if stage == 'early':
            # Show 1 to 3 early symptoms
            num_symptoms = random.randint(1, 3)
            symptoms_to_show.extend(random.sample(early_symptoms, num_symptoms))
            
        elif stage == 'furious':
            # Show all early symptoms plus 2 to 4 furious symptoms
            symptoms_to_show.extend(early_symptoms)
            num_furious = random.randint(2, 4)
            symptoms_to_show.extend(random.sample(furious_symptoms, num_furious))

        elif stage == 'paralytic':
            # Show a full range of symptoms from all stages
            symptoms_to_show.extend(early_symptoms)
            symptoms_to_show.extend(random.sample(furious_symptoms, random.randint(1, 3)))
            symptoms_to_show.extend(random.sample(paralytic_symptoms, random.randint(1, 3)))
            
        for symptom in set(symptoms_to_show): # Use set to avoid duplicates
            df.loc[index, symptom] = 'yes'
            
    # Add realistic "noise" for non-rabid dogs
    non_rabid_indices = df[df['rabies'] == 'no'].index
    # 5% of healthy dogs might have one non-specific, early symptom
    for _ in range(int(len(non_rabid_indices) * 0.05)): 
        dog_index = random.choice(non_rabid_indices.tolist())
        # Noise should only be common, non-specific symptoms
        symptom_to_show = random.choice(early_symptoms) 
        df.loc[dog_index, symptom_to_show] = 'yes'
        
    return df

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Generating {NUM_RECORDS} records with realistic symptoms...")
    
    # 1. Create base data
    main_df = generate_base_data(NUM_RECORDS)
    
    # 2. Assign rabies status
    main_df = assign_rabies(main_df)
    
    # 3. Generate symptoms based on rabies status
    main_df = generate_symptoms(main_df)
    
    # 4. Save to CSV
    main_df.to_csv(OUTPUT_FILENAME, index=False)
    
    print(f"Dataset successfully generated and saved as '{OUTPUT_FILENAME}'")
    print("\n--- Dataset Preview (showing symptom columns) ---")
    preview_cols = ['dog_breed', 'vaccinated', 'rabies'] + main_df.columns.tolist()[-10:]
    print(main_df[preview_cols].head())
    
    print("\n--- Rabies Distribution ---")
    print(main_df['rabies'].value_counts())

    print("\n--- Example of a Rabid Dog's Symptoms ---")
    print(main_df[main_df['rabies'] == 'yes'].iloc[0:1][preview_cols])

    print("\n--- Example of a Non-Rabid Dog's Symptoms (potential noise) ---")
    print(main_df[(main_df['rabies'] == 'no') & (main_df[main_df.columns[-10:]] == 'yes').any(axis=1)].head(1)[preview_cols])