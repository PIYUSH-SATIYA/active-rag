import json
from src.models.qwen import QwenModel
from src.flare.agent import FLAREAgent
from tests.test_flare import MockRetriever


def main():
    print("Loading config...")
    with open("configs/2wikihop_qwen_flare_config.json", "r") as f:
        config = json.load(f)

    print("Loading Qwen model... (this may take a while if using GPU)")
    # If testing without GPU, we can use a mock or a smaller model.
    # For now we use from_pretrained with quantize_4bit if GPU is limited.
    # We'll use device_map="auto".
    try:
        generator = QwenModel.from_pretrained(quantize_4bit=True)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Falling back to MockGenerator for testing...")
        from tests.test_flare import MockGenerator
        from src.models.qwen import GeneratorResult
        generator = MockGenerator([
            GeneratorResult(text="This is a test answer.", token_ids=[1,2,3], tokens=["This", " is", " a"], probabilities=[0.9, 0.9, 0.9])
        ])

    print("Initializing Mock Retriever...")
    retriever = MockRetriever()

    agent = FLAREAgent(generator=generator, retriever=retriever, config=config)

    question = "What is the primary function of FLARE?"
    print(f"\nQuestion: {question}")
    
    result = agent.generate(question)
    
    print("\n--- Final Answer ---")
    print(result.text)
    
    print("\n--- Generation History ---")
    print(json.dumps(result.generation_history, indent=2))
    
    print("\n--- Retrieval History ---")
    print(json.dumps(result.retrieval_history, indent=2))
    
    print("\n--- Confidence History ---")
    print(json.dumps(result.confidence_history, indent=2))


if __name__ == "__main__":
    main()
