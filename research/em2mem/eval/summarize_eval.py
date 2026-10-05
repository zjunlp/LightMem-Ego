import json
import glob

OUTPUT_FILE = "output/videomme/gpt_5_mini_gpt_5/videomme_eval.json"
GPU_FILES = glob.glob("output/videomme/gpt_5_mini_gpt_5/videomme_eval_long_gpu*.json")

def summarize_and_calculate_accuracy():
    data = {}
    for gpu_file in GPU_FILES:
        with open(gpu_file, 'r', encoding='utf-8') as f:
            gpu_data = json.load(f)
            data.update(gpu_data)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    for gpu_file in GPU_FILES:
        os.remove(gpu_file)

    processed = 0
    correct = 0
    for video_id, content in data.items():
        processed += content.get('processed', 0)
        correct += content.get('correct', 0)

    print(f"total: {processed}, correct: {correct}, accuracy: {correct/processed:.2%}")

if __name__ == "__main__":
    summarize_and_calculate_accuracy()