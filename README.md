# Planning RAG

A repository demonstrating how to **decompose complex queries into multiple retrieval steps (Planning) rather than trying to handle a single monolithic query**. This approach is especially useful for Retrieval-Augmented Generation (RAG) workflows, where multi-step planning leads to more precise document retrieval and better final answers.

---

## Why a Planning-Based Approach for RAG?

When dealing with **complex or ambiguous queries**, a naive approach might be to send one large query to the retrieval system. However, this can lead to:
- **Broad or imprecise search results**: A single, overly general query might fetch irrelevant documents.
- **Higher computational costs**: Attempting to handle all sub-topics at once can overload the retrieval pipeline and the subsequent generation step.

### Benefits of Multi-Step Planning

1. **Targeted Sub-Queries**  
   By breaking down a complex query into smaller sub-queries, you can retrieve more accurate and relevant documents at each step.

2. **Iterative Refinement**  
   You can refine the search based on intermediate findings. If the first sub-query reveals certain keywords or insights, the next sub-query can leverage that knowledge for more focused retrieval.

3. **Improved Relevance**  
   Each step can filter out noise and accumulate evidence, leading to a final result that is more coherent and grounded in the retrieved facts.

4. **Scalability and Modularity**  
   If certain steps are repeated or often needed (e.g., “fetch references about person X” or “find definitions for domain-specific terms”), a planning approach can encapsulate these steps into reusable functions or prompts.

---

## Repository Overview

- **Planning RAG Examples**: Scripts and examples showing how to use multi-step queries in a RAG pipeline.
- **PlanningDataset**: A dataset format that can store both complex queries and their sub-queries (plans) for training or demonstration.
- **Inference Pipeline**: Code to demonstrate how an LLM or a simple question-answering system can leverage planning to iteratively query your retrieval backend.

---

## Example Workflow

1. **Initial Query**  
   The user asks: “How do I organize a scientific conference, including budgeting, speaker invitations, and venue logistics?”

2. **Planning**  
   Instead of sending this entire query as-is to your retrieval system, break it down into smaller sub-questions:
   - **Sub-Query 1**: “What are the typical budget items for organizing a scientific conference?”  
   - **Sub-Query 2**: “How to invite speakers and set up a speaker lineup?”  
   - **Sub-Query 3**: “What are best practices for venue selection and logistics for conferences?”

3. **Iterative Retrieval**  
   1. Retrieve documents related to conference budgeting.  
   2. Retrieve documents related to speaker invitations and scheduling.  
   3. Retrieve documents for venue logistics.

4. **Synthesis**  
   Combine insights from the retrieved documents in each step to produce a comprehensive final answer. An LLM or a rules-based approach can handle the synthesis.

## Experiment

To do...

## Expected Results

By adopting a planning-based RAG approach, you should see:
- More relevant retrieval when handling multifaceted or ambiguous queries.
- Clearer, structured answers as the final output—since each sub-query addresses a specific aspect of the overall problem.
- Easier debugging and transparency—each retrieval step is isolated, making it simpler to identify which sub-query or retrieval step might need improvement.

---

Feel free to adapt this workflow to your own research or production needs. We hope you find this approach beneficial for more efficient, accurate, and explainable retrieval-augmented generation!

