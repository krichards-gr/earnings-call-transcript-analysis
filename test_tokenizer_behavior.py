from transformers import AutoTokenizer, pipeline
import os

def test_tokenizer():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "deberta-v3-base-absa-v1.1")
    
    print(f"Loading local tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    text = "The company reported strong earnings."
    pair = "earnings"
    
    print("\n--- TEST 1: Broken Scenario (passing list of pairs as kwarg with single text) ---")
    try:
        # What happens when pipeline iterates: one 'text' item, but 'text_pair' is passed as the WHOLE list via kwargs?
        # tokenizer(text, text_pair=[pair, pair]) -> Should FAIL or produce weird output
        print("Calling tokenizer with text and list of pairs as text_pair kwarg...")
        res = tokenizer(text, text_pair=[pair, pair])
        input_ids = res['input_ids']
        print(f"Result tokens length: {len(input_ids)}")
        print("Result (decoded):", tokenizer.decode(input_ids))
        print("Wait, this WORKED? Did it encode 'text' + 'pair1' + 'pair2'?")
    except TypeError as e:
        print(f"FAILED as expected with TypeError: {e}")
    except Exception as e:
        print(f"FAILED with unexpected error: {e}")

    print("\n--- TEST 2: Proposed Fix 1 (Zip into list of tuples) ---")
    try:
        # tokenizer([list of tuples]) -> Batch encoding
        batch_input = [(text, pair), (text, pair)]
        print(f"Calling tokenizer(batch_input) where batch_input is {batch_input}...")
        res = tokenizer(batch_input, padding=True, truncation=True, return_tensors="pt")
        print(f"Result shape: {res['input_ids'].shape}")
        print("Result (decoded 0):", tokenizer.decode(res['input_ids'][0]))
        print("SUCCESS: Tokenizer accepts list of tuples for batch processing of pairs.")
    except Exception as e:
        print(f"FAILED: {e}")

    print("\n--- TEST 3: Pipeline behavior check ---")
    try:
        # If we use pipeline, can we pass a list of tuples?
        # We can simulate what pipeline likely does internally if we pass just 'inputs' as list of tuples
        # pipeline iteration: item = (text, pair)
        # item passed to process -> tokenizer(item) -> tokenizer((text, pair))
        
        # Test tokenizer with a SINGLE tuple
        single_item = (text, pair)
        print(f"Calling tokenizer(single_item) where single_item is {single_item}...")
        res = tokenizer(single_item) # Does tokenizer((A,B)) treat it as pair?
        # Note: tokenizer((A,B)) might be treated as batch of 2 separate sentences?
        # Or as pair A+B?
        print("Result (decoded):", tokenizer.decode(res['input_ids']))
        # We need to see if it put [SEP] in between.
        
        if tokenizer.sep_token_id in res['input_ids']:
             print("It contains SEP token.")
        else:
             print("It does NOT contain SEP token (might be treated as single sentence?).")

    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    test_tokenizer()
