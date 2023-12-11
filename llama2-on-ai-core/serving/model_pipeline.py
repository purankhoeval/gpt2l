import os
import sys
import torch
import transformers
import huggingface_hub
import gc

transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

HUB_TOKEN = "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"
huggingface_hub.login(token=HUB_TOKEN)

os.environ["TRANSFORMERS_CACHE"] = "shared/IMR/gpt2xl/cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "MAX_SEGMENT_SIZE:1073741824"  # 1 GB, adjust as needed

class Model:
    generator = None

    @staticmethod
    def release_memory():
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def setup():
        print("START LOADING SETUP", file=sys.stderr)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if Model.generator:
            del Model.generator
            Model.release_memory()

        model_name = "EleutherAI/gpt-neo-2.7B"

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        Model.release_memory()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device=device,
            trust_remote_code=True,
            load_in_4bit=True,
            use_auth_token=True,
            batch_size=1
        )
        Model.generator = lambda prompt, args: pipeline(
            prompt,
            **{
                "max_length": 2000,
                "do_sample": True,
                "top_k": 10,
                "num_return_sequences": 1,
                "eos_token_id": tokenizer.eos_token_id,
                **args
            }
        )

        print("SETUP DONE", file=sys.stderr)

    @staticmethod
    def predict(prompt, args):
        with torch.no_grad():
            result = Model.generator(prompt, args)
            Model.release_memory()

            # Update hidden_states to torch.float16
            hidden_states = result[0]["hidden_states"]
            hidden_states = hidden_states.to(torch.float16)

            result[0]["hidden_states"] = hidden_states
            return result

if __name__ == "__main__":
    Model.setup()
    print(Model.predict("Hello, who are you?", {}))
