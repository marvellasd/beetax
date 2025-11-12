import os
import argparse
import json
from ragas import evaluate
from langchain.chat_models import ChatOpenAI
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
)
from dotenv import load_dotenv
from datasets import Dataset
load_dotenv()

class Evaluator():
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def evaluate(self,filepath, outputpath):
        with open(filepath, "r", encoding= "utf-8") as f:
            sample_data = json.load(f)

        sample_data = Dataset.from_list(sample_data)

        eval_metrics = [
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_precision,
            context_recall,
        ]

        results = evaluate(llm=self.llm, dataset=sample_data, metrics=eval_metrics)
        try:
            results_dict = results.dict()
        except AttributeError:
            results_dict = results.__dict__
        
        with open(os.path.join(outputpath, "evaluation_result.json"), "w", encoding="utf-8") as f:
            json.dump(results_dict['_repr_dict'], f, indent=4, ensure_ascii=False) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on a dataset.")
    parser.add_argument(
        "--filepath", 
        type=str, 
        required=True, 
        help="Path to the JSON file containing the sample data."
    )
    parser.add_argument(
        "--outputpath", 
        type=str, 
        required=True, 
        help="Path to the JSON file containing the result data."
    )
    args = parser.parse_args()

    evaluator = Evaluator()
    results = evaluator.evaluate(args.filepath, args.outputpath)