import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

def load_codestral():
    model_path = r"your\file\path" # filepath to model (default $HOME/mistral_models/Codestral-22B-v0.1)
    
    print("Loading Codestral model...")
    
    # Configure 4-bit quantization (optional)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load config from the model's config.json
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    tokenizer = MistralTokenizer.v3()
    
    # Load model with quantization from local path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    print("Model loaded successfully!")
    return model, tokenizer

def main():
    try:
        model, tokenizer = load_codestral()
        
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            user_input = input("\nInput: ")
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif not user_input:
                continue
            
            print("\nOutput: ", end="", flush=True)
            
            try:
                request = ChatCompletionRequest(messages=[UserMessage(content=user_input)])
                tokens = tokenizer.encode_chat_completion(request).tokens
                input_length = len(user_input) + 1
                # Convert list to tensor and move to GPU
                input_ids = torch.tensor([tokens], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                generated_ids = model.generate(
                    input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=1000, 
                    do_sample=True,
                    pad_token_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
                )
                response = tokenizer.decode(generated_ids[0].tolist())
                response = response[input_length:]
                print(response)
            except Exception as e:
                print(f"Error generating response: {e}")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model path is correct and all files are present.")

if __name__ == "__main__":
    main() 
