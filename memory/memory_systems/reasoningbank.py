import os
import json
import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
# from google import genai
# from google.genai.types import HttpOptions, GenerateContentConfig
# from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from openai import OpenAI
from datetime import datetime
import uuid
import tiktoken

# Config Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# System Instructions for Memory Extraction, depending on success or failure

SUCCESSFUL_SI = """
You are a memory expert. You will be given a task query to be completed, the corresponding trajectory that represents **how an agent successfully solved the task**. 

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully resolve the issue in the future>
```
"""

FAILED_SI = """
You are a memory expert. You will be given a task query to be completed, the corresponding trajectory that represents **how an agent attempted to resolve the task but failed**. 

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully resolve the issue in the future>
```
"""


class ReasoningBankMemorySystem:
    def __init__(
        self, 
        storage_path: str = None,
        embedding_path: str = None,
        model_name: str = None,
        embedding_model: str = "text-embedding-3-small",
        embed_instruction: str= "Given the prior task queries, your task is to analyze a current query's intent and select relevant prior queries that could help complete it.",
        topk=5,
        user_id:  Optional[str] = None, 
       
    ):
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        # 落盘默认走 memory.state（上游原值 "./reasoningbank_data/" 是 server 的 cwd，
        # 会被 git clean 抹掉）。模型默认跟上游一致；MARENA_REASONINGBANK_MODEL 可以
        # 把它拉齐到 agent 侧——两侧模型不同会在对比里多出一个变量。
        from memory.state import state_dir
        base = str(state_dir("reasoningbank")) + os.sep
        self.storage_path = storage_path or base
        self.embedding_path = embedding_path or base
        self.model_name = model_name or os.getenv("MARENA_REASONINGBANK_MODEL") or "gpt-4.1-mini"
        self.embedding_model_name = embedding_model
        self.embed_instruction = embed_instruction
        self.topk=topk
        # initialize LLM client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        
        self.user_id=user_id
        self.storage_path=os.path.join(self.storage_path , f"{self.user_id}_reasoning_bank.jsonl")
        self.embedding_path=os.path.join(self.embedding_path , f"{self.user_id}_embeddings.jsonl")
        # ensure memory directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)
        
        # load existing memory bank and embeddings
        self.memory_bank = self._load_jsonl(self.storage_path)


    def _load_jsonl(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
        

    def _l2_normalize(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return F.normalize(x, p=2, dim=dim)
    
    def wrap_user_prompt(self, prompt: str):
        memory_context_lines = ["<memory_context>"]

        results= self.retrieve(prompt, n=self.topk)
        if results:
            for chunk in results:
                memory_context_lines.append(f"<memory>{chunk}</memory>")
        else:
            memory_context_lines.append("None")

        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User: {prompt}")
        return "\n".join(memory_context_lines)

    def add_chunk(self, chunk: str):
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        return self.add(
            timestamp=timestamp,
            instance_id=str(uuid.uuid4()),
            content=chunk,
        )
        

     
    
    def _get_openai_embedding(self, text: str, maxlen: int = 4096) -> torch.Tensor:
        """Obtain text embedding using Gemini Text Embedding Model."""
        resp = self.client.embeddings.create(
            model=self.embedding_model_name,
            input=[text[:maxlen]], #4096 letters -> 400 tokens
        )
        return torch.tensor([resp.data[0].embedding], dtype=torch.float32)
        
        
    def _get_qwen_embedding(self, query: str) -> Tuple[torch.Tensor, str, int]:
        """Returns (1, D) torch tensor (on CPU), model_name, dim."""
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left')
        model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        batch = tokenizer([query], max_length=1024, padding=True, truncation=True, return_tensors='pt')
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
            last_hidden = out.last_hidden_state  # (1, L, D)
            masked = last_hidden.masked_fill(~batch['attention_mask'][..., None].bool(), 0.0)
            pooled = masked.sum(dim=1) / batch['attention_mask'].sum(dim=1)[..., None]  # (1, D)
        pooled = pooled.to('cpu')
        pooled = self._l2_normalize(pooled, dim=1)
        
        return pooled
    

    def _load_cached_embeddings(self) -> Tuple[List[str], List[str], torch.Tensor]:
        ids, texts, vecs = [], [], []
        if not os.path.exists(self.embedding_path):
            return ids, texts, torch.empty(0)

        with open(self.embedding_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                ids.append(obj["id"])
                texts.append(obj.get("text", ""))
                vecs.append(obj["embedding"])

        if not vecs:
            return ids, texts, torch.empty(0)

        emb = torch.tensor(vecs, dtype=torch.float32)
        return ids, texts, self._l2_normalize(emb, dim=1)
    

    def _llm_judge_status(self,  trajectory: str) -> bool:
        # print("judgeing status...")
        prompt = f"Task and Trajectory:\n{trajectory}\n\nDid the agent successfully complete the task? Answer with 'success' or 'fail' only."
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
            {"role": "system", "content": "You are a helpful assistant that judges whether the agent successfully completed the task."},
            {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_completion_tokens=128,
        )
        return "success" in response.choices[0].message.content.strip().lower()


    def add(self, timestamp, content: str, instance_id: str):
        """
        Evaluates task performance, generates structured memory items using 
        status-specific instructions, and persists data to both storage and vector cache.

        Args:
            instance_id (str): Unique identifier for the task instance.
            task (str): The original problem statement or user query.
            trajectory (str): The full sequence of actions and thoughts from the agent.
        """
        # 1. Judge the outcome of the trajectory (Success vs. Failure)
        is_successful = self._llm_judge_status(content)
        status_str = "success" if is_successful else "fail"
        
        # 2. Prepare the prompt and select the appropriate System Instruction
        # The prompt combines the original task and the execution history
        generation_prompt = f"**Query and Trajectory:**\n{content}"
        system_instruction = SUCCESSFUL_SI if is_successful else FAILED_SI

        # 3. Generate structured memory items via the LLM
        # This extracts generalizable insights (titles, descriptions, contents)
        generated_memory_items = self._extract_memory_items(
            prompt=generation_prompt, 
            si=system_instruction
        )

        # 4. Persist the complete record to the primary JSONL storage
        record = {
            "timestamp": timestamp,
            "task_id": instance_id, 
            "memory_items": generated_memory_items,
            "status": status_str,
        }
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.memory_bank.append(record)

        # 5. Generate and store the embedding for semantic retrieval
        # We embed the original 'task' to allow matching similar future queries
        if self.embedding_model_name == "text-embedding-3-small":
            q_vec = self._get_openai_embedding(content)
        elif self.embedding_model_name.startswith("Qwen"):
            q_vec = self._get_qwen_embedding(content)
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model_name}")
        
        emb_record = {
            "id": instance_id,
            "text": content,
            "embedding": q_vec.squeeze(0).tolist(),
        }
        with open(self.embedding_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(emb_record) + "\n")
        
        logger.info(f"Successfully indexed {status_str} memory for task: {instance_id}")


    def _extract_memory_items(self, prompt: str, si: str) -> List[str]:
        """
        Internal helper to call the generative model with specific instructions.
        
        Returns:
            List[str]: A list of memory item strings parsed from the LLM response.
        """
        # print("Generating Memory")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
            {"role": "system", "content": si.strip()},
            {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=4096,
        )
        # print("Generated Memory")
        # Split by double newlines to separate individual "Memory Item" blocks
        return response.choices[0].message.content.strip().split("\n\n")
    

    def retrieve(self, cur_query: str, n: int = 1) -> List[Dict]:
        """
        Performs an instruction-aware semantic search to find the top-n most 
        relevant past experiences from the reasoning bank.

        Args:
            cur_query (str): The current task or problem statement to find matches for.
            n (int): The number of relevant memories to retrieve. Defaults to 1.

        Returns:
            List[Dict]: A list of the most relevant memory records containing 
                        task IDs, queries, and generated summaries.
        """
        # Return empty list if no memories have been indexed yet

        cache_ids, cache_texts, cache_emb = self._load_cached_embeddings()

        if cache_emb.numel() == 0:
            logger.warning("ReasoningBank is empty. No records available for retrieval.")
            return []

        # 1. Construct an instruction-augmented prompt for better retrieval alignment
        # Uses the domain-specific instruction defined during initialization
        instruction_task = self.embed_instruction
        full_query = f'Instruct: {instruction_task}\nQuery: {cur_query}'
        
        # 2. Get the embedding for the augmented query and normalize it
        instruct_vec = self._get_openai_embedding(full_query)
        instruct_vec = self._l2_normalize(instruct_vec, dim=1)

        # 3. Calculate Cosine Similarity via dot product (since vectors are L2 normalized)
        # Resulting scores shape: (1, N) -> (N,)
        scores = (instruct_vec @ cache_emb.T).squeeze(0) * 100.0
        
        # 4. Rank candidates by similarity score in descending order
        id2score = list(zip(cache_ids, scores.tolist()))
        id2score.sort(key=lambda x: x[1], reverse=True)
        top_ids = [str(item[0]) for item in id2score[:n]]

        # 5. Map the top-ranked IDs back to their full metadata records
        results = []
        for sid in top_ids:
            for item in self.memory_bank:
                if str(item["task_id"]) == sid:
                    results.append(item)
                    break
        
        return results



# Sample Use
    # res = memory.retrieve(task, n=1)
    # if not res:
    #     selected_memory = ""
    # else:
    #     mem_items = []
    #     for item in res:
    #         for i in item["memory_items"]:
    #             mem_items.append(i)
    #     selected_memory = "\n\n".join(mem_items)

    # progress_manager.on_instance_start(instance_id)
    # progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    # agent = None
    # extra_info = None

    # try:
    #     env = get_sb_environment(config, instance)
    #     agent = ProgressTrackingAgent(
    #         model,
    #         env,
    #         progress_manager=progress_manager,
    #         instance_id=instance_id,
    #         **config.get("agent", {}),
    #     )
    #     exit_status, result = agent.run(task, selected_memory=selected_memory)
    # except Exception as e:
    #     logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
    #     exit_status, result = type(e).__name__, str(e)
    #     extra_info = {"traceback": traceback.format_exc()}
    # finally:
    #     save_traj(
    #         agent,
    #         instance_dir / f"{instance_id}.traj.json",
    #         exit_status=exit_status,
    #         result=result,
    #         extra_info=extra_info,
    #         instance_id=instance_id,
    #         print_fct=logger.info,
    #     )
    #     update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
    #     progress_manager.on_instance_end(instance_id, exit_status)

    #     # read trajectory and extract memory
    #     with open(instance_dir / f"{instance_id}.traj.json", "r") as f:
    #         messages = json.load(f)["messages"]
    #     trajectory = "\n".join([m["content"] for m in messages if m["role"] != "system"])

    #     memory.add(trajectory, task, instance_id)