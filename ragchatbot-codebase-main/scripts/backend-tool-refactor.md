Refactor @backend/ai_generator. py to support sequential tool calling where Claude can make up to 2 tool calls in separate API rounds.


Current behavior:
- Claude makes 1 tool call - tools are removed from API params → final response
- If Claude wants another tool call after seeing results, it can't (gets empty response)



Desired behavior:
- Each tool call should be a separate API request where Claude can reason about previous results