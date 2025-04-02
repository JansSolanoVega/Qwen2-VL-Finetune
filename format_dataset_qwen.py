from datasets import load_from_disk
import json
import random
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import gc
from PIL import Image

# Dictionary to store image files based on id
IMAGE_FILES = {}
RESPONSE_KEYS = ["age", "gender", "anxiety_level", "depression_level", "depressed_at_baseline", "self_harm_ever", "not_worth_living", "neuroticism"]
DATA_STATISTICS = {key: {} for key in RESPONSE_KEYS}
PROMPT_STATISTICS = {}

### Mapping functions
def map_depression(value):
    match(value):
        case value if value < 4:
            return "None"
        case value if 5 <= value <= 9:
            return "Mild"
        case value if 10 <= value <= 14:
            return "Moderate"
        case value if 15 <= value <= 19:
            return "Moderately severe"
        case value if value > 20:
            return "Severe"
        case _:
            return None

def map_anxiety(value):
    match(value):
        case value if value < 5:
            return "None"
        case value if 5 <= value <= 9:
            return "Mild"
        case value if 10 <= value <= 14:
            return "Moderate"
        case value if 15 <= value:
            return "Severe"
        case _:
            return None

def map_int_value(value):
    return int(value) if not pd.isna(value) else None

def map_binary_value(value):
    return "Yes" if value == 1 else "No" if value == 0 else None

def map_recording(value):
    return np.array(value)

def map_gender(value):
    return "Male" if value == 0 else "Female" if value == 1 else None

ORIG_TO_NEW_KEY_MAPPING = {
    "Voxelwise_RobustScaler_Normalized_Recording": {"map_function": map_recording, "new_key": "recording"},
    "Age.At.MHQ": {"map_function": map_int_value, "new_key": "age"},
    "Gender": {"map_function": map_gender, "new_key": "gender"},
    "GAD7.Severity": {"map_function": map_anxiety, "new_key": "anxiety_level"},
    "PHQ9.Severity": {"map_function": map_depression, "new_key": "depression_level"},
    "Depressed.At.Baseline": {"map_function": map_binary_value, "new_key": "depressed_at_baseline"},
    "Self.Harm.Ever": {"map_function": map_binary_value, "new_key": "self_harm_ever"},
    "Not.Worth.Living": {"map_function": map_binary_value, "new_key": "not_worth_living"},
    "Neuroticism": {"map_function": map_int_value, "new_key": "neuroticism"}
}

def map_function(example):
    for key, values in ORIG_TO_NEW_KEY_MAPPING.items():
        if key in example:
            example[values["new_key"]] = values["map_function"](example[key])
    return example

### Load and format data
def load_prompts(prompts_file):
    """Load prompts from JSON file."""
    with open(prompts_file, "r") as f:
        return json.load(f)

def format_response(response_template, sample_data, response_vars):
    """Format response string using values from sample data."""
    response_dict = {var: str(sample_data[var]) for var in response_vars}
    return response_template.format(**response_dict)

def process_and_filter_sample(example):
    """Process a single sample and check if it has valid responses."""
    processed = {}
    for key, values in ORIG_TO_NEW_KEY_MAPPING.items():
        if key in example:
            processed[values["new_key"]] = values["map_function"](example[key])
    
    # Check if sample has all valid responses
    has_valid = all(
        processed.get(key) is not None
        for key in RESPONSE_KEYS
    )
    
    return processed if has_valid else None

def create_in_memory_dataset(dataset, args):
    """Create a filtered, processed dataset in memory."""
    processed_data = []
    print("Processing and filtering dataset in memory...")

    if args.subsample_size:
        dataset = dataset.select(range(args.subsample_size))

    for example in tqdm(dataset, desc="Processing samples"):
        processed = {}
        for key, values in ORIG_TO_NEW_KEY_MAPPING.items():
            if key in example:
                processed[values["new_key"]] = values["map_function"](example[key])
        
        # Check if sample has all valid responses
        has_valid = all(
            processed.get(key) is not None
            for key in RESPONSE_KEYS
        )
        
        if has_valid:
            processed_data.append(processed)
    
    print(f"Processed {len(processed_data)} valid samples")
    return processed_data

def update_statistics(index, response):
    """Update statistics dictionary with sample information for used response variables."""
    for key, value in response.items():
        if key in RESPONSE_KEYS:  # Only track statistics for our key response variables
            if value not in DATA_STATISTICS[key]:
                DATA_STATISTICS[key][value] = []
            DATA_STATISTICS[key][value].append(index)

def update_prompt_statistics(prompt_input):
    """Update prompt statistics dictionary."""
    PROMPT_STATISTICS[prompt_input] = PROMPT_STATISTICS.get(prompt_input, 0) + 1

def print_statistics():
    """Print statistics dictionary."""
    for key in RESPONSE_KEYS:
        if key in DATA_STATISTICS:
            print(f"\n{key}:")
            for value, sample_ids in DATA_STATISTICS[key].items():
                print(f"  {value}: {len(sample_ids)}")

def save_statistics(output_dir, statistics, filename="statistics.json"):
    """Save statistics dictionary to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / filename, "w") as f:
        json.dump(statistics, f, indent=2)

def create_formatted_dataset(prompts, processed_data, num_samples, images_dir):
    """Create formatted dataset from pre-processed data."""
    formatted_data = []
    images_dir = Path(images_dir)
    images_dir.mkdir(exist_ok=True, parents=True)
    
    sample_count = 0
    dataset_length = len(processed_data)
    progress_bar = tqdm(total=num_samples, desc="Formatting dataset...")
    prompt = prompts[0]

    for sample_idx in range(dataset_length):
        sample = processed_data[sample_idx]
        if all(
                var in sample and sample[var] is not None
                for var in prompt["response_vars"]
            ):
                valid_sample = sample
                filename = f"{sample_idx}.png"
                recording = valid_sample["recording"]
                normalized_recording = (255 * (recording - np.min(recording)) / (np.max(recording) - np.min(recording))).astype(np.uint8)
                image = Image.fromarray(normalized_recording)
                image.save(str(images_dir / filename))

                try:
                    if prompt["response"][:2] != "{{":
                        response_template = "{" + prompt["response"] + "}"
                    else:
                        response_template = prompt["response"]

                    response = format_response(
                        response_template,
                        valid_sample,
                        prompt["response_vars"]
                    )
                except Exception as e:
                    print(f"Error formatting response: {e}")
                    continue

                # Construct the formatted sample
                formatted_sample = {
                    "id": f"{sample_count:012d}",
                    "image": filename,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\n"+prompt["model_input"]
                        },
                        {
                            "from": "gpt",
                            "value": response
                        }
                    ]
                }
                formatted_data.append(formatted_sample)
                sample_count += 1
                progress_bar.update(1)
                if sample_count==num_samples:
                    break
        else:
            pass

    print(f"Processed {sample_count} samples")
    return formatted_data

def balance_dataset(data, max_iterations=1000):
    """Balance dataset by oversampling underrepresented classes.
    
    Args:
        data: List of formatted samples
    
    Returns:
        List of formatted samples with additional balanced samples
    """
    balanced_data = data.copy()
    target_counts = {}
    
    # Calculate target count for each response variable (using 90% of max count as target)
    for key in RESPONSE_KEYS:
        if DATA_STATISTICS[key]:
            max_count = max(len(samples) for samples in DATA_STATISTICS[key].values())
            target_counts[key] = int(max_count * 0.9)  # Allow 10% variance
    
    iteration = 0
    made_progress = True
    
    while made_progress and iteration < max_iterations:
        made_progress = False
        iteration += 1
        
        # Find most underrepresented class across all response variables
        min_ratio = float('inf')
        target_var = None
        target_value = None
        
        for var in RESPONSE_KEYS:
            if var not in target_counts or target_counts[var] == 0:
                continue
            
            for value, samples in DATA_STATISTICS[var].items():
                ratio = len(samples) / target_counts[var]
                if ratio < min_ratio:
                    min_ratio = ratio
                    target_var = var
                    target_value = value
        
        # Stop if we've reached at least 90% of target counts for all classes
        if min_ratio >= 0.9:
            break

        # Get samples that have the target value
        candidate_samples = []
        for sample_index in DATA_STATISTICS[target_var][target_value]:
            if sample_index < len(data):
                candidate_samples.append(data[sample_index])
        
        if not candidate_samples:
            continue
            
        # Randomly select a sample to duplicate
        sample_to_duplicate = random.choice(candidate_samples)
        
        # Create new sample with new ID
        new_sample = sample_to_duplicate.copy()
        new_id = f"{len(balanced_data):012d}"
        new_sample["id"] = new_id

        # Update statistics for all response variables in this sample
        update_statistics(len(balanced_data), new_sample["response"])
        update_prompt_statistics(new_sample["messages"][0]["content"][0]["text"])

        balanced_data.append(new_sample)
        made_progress = True
    
    print(f"\nBalancing completed after {iteration} iterations")
    print("Final class distribution:")
    print_statistics()

    print(f"Final dataset length: {len(balanced_data)}")
    return balanced_data

def save_dataset(data, output_dir):
    """Save formatted dataset to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / "formatted_dataset.json", "w") as f:
        json.dump(data, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Format dataset with specified paths")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input dataset"
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        required=True,
        help="Path to the prompts file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the formatted dataset and statistics"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory to save the image files"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to format"
    )
    parser.add_argument(
        "--subsample_size",
        type=int,
        default=None,
        help="Number of data points to subsample to prior to formatting"
    )
    parser.add_argument(
        "--balance_iterations",
        type=int,
        default=1000,
        help="Maximum number of iterations to try to balance the dataset"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load raw dataset
    dataset = load_from_disk(args.data_path)
    
    # Load prompts
    prompts = load_prompts(args.prompts_path)
    
    # Create in-memory processed dataset
    processed_data = create_in_memory_dataset(dataset, args)

    del dataset
    gc.collect()
    
    # Create formatted dataset from processed data
    formatted_data = create_formatted_dataset(
        prompts=prompts,
        processed_data=processed_data,
        num_samples=args.num_samples,
        images_dir=args.images_dir
    )

    save_statistics(args.output_dir, DATA_STATISTICS, "pre_balancing_statistics.json")
    save_statistics(args.output_dir, PROMPT_STATISTICS, "pre_balancing_prompt_statistics.json")
    
    # Balance the dataset
    balanced_data = balance_dataset(formatted_data, args.balance_iterations)

    save_statistics(args.output_dir, DATA_STATISTICS, "post_balancing_statistics.json")
    save_statistics(args.output_dir, PROMPT_STATISTICS, "post_balancing_prompt_statistics.json")
    
    # Save balanced dataset
    save_dataset(balanced_data, args.output_dir)

if __name__ == "__main__":
    main()