A/B testing professionals need a trusted, reliable source of truth for deep A/B testing questions and reports, without spending all day grinding through papers.

This is a problem for data driven product managers, A/B testing data scientists, and experimentation managers. They come across deep A/B testing questions at work that simple google searches or even Perplexity searches will genuinely not solve. I’ve been in these shoes before. I remember searching and trying to figure out this particular deep A/B testing question:

“Suppose we want to run an A/B test with a few variants (say 3-4 total) for some ML models and we are triggering and logging the counterfactuals. Even if we had each variant log all the other counterfactuals (which would make Control and variants on an 'even playing field'), it seems like the slow down would hurt the business overall. Any practical advice or pro-tips on how to deal with this situation?”

I spent a lot of time parsing through different sources, not knowing which I could trust, not clearly understanding what they were saying on first pass. This app solves this problem. Ron Kohavi is a trusted source of truth when it comes to A/B testing. At the same time, his book is not available on the internet. I personally asked him if I could use his book, papers, and LinkedIn posts (which I scrapped!) to form the database for this app. Now data driven product managers, A/B testing data scientists, and experimentation managers have a reliable and convenient source of truth for their deep A/B testing questions.

Users will be able to ask the app any A/B testing question. The app then first uses RAG to try to answer the question using the Kohavi collection. An LLM judge determines if the response is good enough or not. If not, state moves to the agent, which has access to 3 tools: this RAG node again, Arxiv, and this RAG node but with an LLM rephrasing the query. Finally, when the app is ready to answer the question, it first generates 3 follow up questions, then responds to the user with the answer, the sources (including title and page number), and 3 follow up questions. The user can then proceed to ask more questions.

Furthermore, by clicking on the toggle on the left-side panel, users can switch to report-generating mode, which creates deep research reports on any topic within A/B testing. For each section of the report that needs research, it will first try to get the information it needs through the same Kohavi collection. If it can't properly write the section using just this collection, it will then search arXiv. If that's not enough, it will finally search the web using Tavily. It provides ALL sources, section by section. Both modes have been trained to only respond based on the sources they retrieve.

Presentation slides can be found here:

https://docs.google.com/presentation/d/1HKNfEHjNoKetoRHbQ8ijlEXqy1ugi4V6NHvVeiqhgGM/edit?usp=sharing 

The actual app (v3) can be found here:

https://huggingface.co/spaces/kamkol/AB_AI_v3 