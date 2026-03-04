import json
import os


class DocFusionSolution:

  def train(self, train_dir: str, work_dir: str) -> str:
    """
    Dummy training step.
    Just creates a folder for model artifacts.
    """

    model_dir = os.path.join(work_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    meta = {
      "version": "baseline",
      "note": "initial scaffold"
    }

    with open(os.path.join(model_dir, "meta.json"), "w") as f:
      json.dump(meta, f)

    return model_dir


  def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
    """
    Dummy prediction step.
    Outputs valid predictions with no extraction yet.
    """

    test_json = os.path.join(data_dir, "test.jsonl")

    predictions = []

    with open(test_json, "r") as f:
      for line in f:
        record = json.loads(line)

        predictions.append({
          "id": record["id"],
          "vendor": None,
          "date": None,
          "total": None,
          "is_forged": 0
        })

    with open(out_path, "w") as f:
      for p in predictions:
        f.write(json.dumps(p) + "\n")