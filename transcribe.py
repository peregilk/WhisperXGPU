import argparse
import whisperx
import gc
import torch
from datasets import load_dataset
import numpy as np
import jsonlines
import os
from datetime import datetime
import json
import time
from google.cloud import storage

def make_serializable(item):
    """Convert ndarray objects to lists to ensure JSON serializability."""
    for key, value in item.items():
        if isinstance(value, np.ndarray):
            item[key] = value.tolist()
        elif isinstance(value, dict):
            item[key] = make_serializable(value)
    return item

def process_batch(batch, model, model_a, metadata, language, batch_size, device):
    results = []
    for item in batch:
        audio_array = np.array(item['audio']['array']).astype(np.float32)
        result = model.transcribe(audio_array, batch_size=batch_size, language=language)
        raw_transcripts = [segment['text'] for segment in result['segments']]
        
        aligned_result = whisperx.align(result["segments"], model_a, metadata, audio_array, device, return_char_alignments=False)
        transcripts_with_timestamps = aligned_result["segments"]
        wav2vec_transcripts = [segment['text'] for segment in transcripts_with_timestamps]

        results.append((raw_transcripts, transcripts_with_timestamps, wav2vec_transcripts))
    return results

def upload_to_bucket(bucket_name, file_path, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)
    print(f"Uploaded {file_path} to gs://{bucket_name}/{blob_name}")

def print_items_from_shard_and_transcribe(dataset_name, split, num_shards, shard_indices, max_samples, device, batch_size, language, model_name, output_dir, bucket):
    # Load the dataset in streaming mode with trust_remote_code=True
    dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
    
    # If shard_indices is not provided, read everything
    if shard_indices:
        shard_indices = list(map(int, shard_indices.split(',')))
        # Create a filter to skip items not in the specified shards
        def shard_filter(dataset, num_shards, shard_indices):
            return (item for i, item in enumerate(dataset) if i % num_shards in shard_indices)
        # Apply the filter to get the specific shards
        filtered_dataset = shard_filter(dataset, num_shards, shard_indices)
    else:
        filtered_dataset = dataset

    # Create an iterator
    iterator = iter(filtered_dataset)

    # Load the WhisperX model
    model = whisperx.load_model(model_name, device, compute_type="float16")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file name
    shard_str = '_'.join(map(str, shard_indices)) if shard_indices else 'all_shards'
    base_filename = f"{dataset_name.replace('/', '_')}_shards_{shard_str}"
    output_file = os.path.join(output_dir, f"{base_filename}.jsonl")
    summary_file = os.path.join(output_dir, f"summary_{base_filename}.json")

    # Initialize statistics
    start_time = time.time()
    total_samples = 0

    with jsonlines.open(output_file, mode='w') as writer:
        batch = []
        for i, item in enumerate(iterator):
            if max_samples is not None and i >= max_samples:
                break

            batch.append(item)
            total_samples += 1

            if len(batch) >= batch_size:
                results = process_batch(batch, model, model_a, metadata, language, batch_size, device)

                for j, batch_item in enumerate(batch):
                    # Convert the item to a JSON serializable format
                    item_serializable = make_serializable(batch_item)

                    # Prepare output dictionary
                    raw_transcripts, transcripts_with_timestamps, wav2vec_transcripts = results[j]
                    output = {
                        "dataset_name": dataset_name,
                        "shard_indices": shard_indices,
                        "whisper_model_name": model_name,
                        "language": language,
                        "datetime": datetime.now().isoformat(),
                        "input": item_serializable,
                        "whisperx": {
                            "raw_transcripts": raw_transcripts,
                            "transcripts_with_timestamps": transcripts_with_timestamps,
                            "wav2vec_transcripts": wav2vec_transcripts
                        }
                    }

                    # Write to the output file
                    writer.write(output)

                # Upload to bucket every 1000 lines (approximately)
                if bucket and total_samples % 100 < batch_size:
                    upload_to_bucket(bucket, output_file, f"{base_filename}.jsonl")

                batch = []
                # Clear GPU memory if necessary
                gc.collect()
                torch.cuda.empty_cache()

        # Process any remaining items in the last batch
        if batch:
            results = process_batch(batch, model, model_a, metadata, language, batch_size, device)

            for j, batch_item in enumerate(batch):
                # Convert the item to a JSON serializable format
                item_serializable = make_serializable(batch_item)

                # Prepare output dictionary
                raw_transcripts, transcripts_with_timestamps, wav2vec_transcripts = results[j]
                output = {
                    "dataset_name": dataset_name,
                    "shard_indices": shard_indices,
                    "whisper_model_name": model_name,
                    "language": language,
                    "datetime": datetime.now().isoformat(),
                    "input": item_serializable,
                    "whisperx": {
                        "raw_transcripts": raw_transcripts,
                        "transcripts_with_timestamps": transcripts_with_timestamps,
                        "wav2vec_transcripts": wav2vec_transcripts
                    }
                }

                # Write to the output file
                writer.write(output)

            # Clear GPU memory if necessary
            gc.collect()
            torch.cuda.empty_cache()

    # Upload the final output to the bucket
    if bucket:
        upload_to_bucket(bucket, output_file, f"{base_filename}.jsonl")

    # Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time
    average_time_per_transcription = total_time / total_samples if total_samples > 0 else 0

    summary = {
        "total_samples": total_samples,
        "datetime": datetime.now().isoformat(),
        "total_time": total_time,
        "average_time_per_transcription": average_time_per_transcription
    }

    # Save summary statistics to file
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

    # Upload the summary to the bucket
    if bucket:
        upload_to_bucket(bucket, summary_file, f"summary_{base_filename}.json")

    # Clean up the model to free up GPU memory
    del model
    del model_a
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print items from a dataset and transcribe using WhisperX.")
    parser.add_argument('--dataset_name', type=str, default='NbAiLab/ncc_speech_v7', help='The name of the dataset to load.')
    parser.add_argument('--split', type=str, default='train', help='The split of the dataset to load (e.g., train, test, validation).')
    parser.add_argument('--num_shards', type=int, default=256, help='The total number of shards.')
    parser.add_argument('--shard_indices', type=str, default=None, help='A comma-separated list of shard indices to load (0-indexed). If not set, all shards will be read.')
    parser.add_argument('--max_samples', type=int, default=None, help='The maximum number of samples to print and transcribe. If not set, it is unlimited.')
    parser.add_argument('--device', type=str, default='cuda', help='The device to use for inference (e.g., "cuda" or "cpu").')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for transcription.')
    parser.add_argument('--language', type=str, default='no', help='The language to use for transcription and alignment.')
    parser.add_argument('--model_name', type=str, default='NbAiLab/nb-whisper-small', help='The name of the Whisper model to use.')
    parser.add_argument('--output_dir', type=str, required=True, help='The directory where the results will be saved.')
    parser.add_argument('--bucket', type=str, default=None, help='The name of the Google Cloud Storage bucket to upload results.')

    args = parser.parse_args()
    
    print_items_from_shard_and_transcribe(
        args.dataset_name, args.split, args.num_shards, args.shard_indices, args.max_samples, args.device, args.batch_size, args.language, args.model_name, args.output_dir, args.bucket
    )

