import re
from gateway.schemas import QueryType


class QueryClassifier:

    PATTERNS = {
        QueryType.CODING: [
            r"\b(code|function|implement|debug|fix|error|class|api|sql|query|script"
            r"|python|javascript|typescript|java|rust|golang|bash|dockerfile|regex"
            r"|algorithm|recursion|loop|variable|compile|syntax|program|method"
            r"|object|array|string|integer|boolean|library|framework|database"
            r"|endpoint|request|response|async|thread|process|memory|pointer)\b",
            r"```",
            r"\b(write a|create a|build a|implement a|develop a)\b.*(function|class|script|program|api|tool)",
        ],
        QueryType.SUMMARIZATION: [
            r"\b(summarize|summary|tldr|brief|shorten|condense|key points|overview"
            r"|highlights|abstract|recap|digest|main points|bullet points)\b",
        ],
        QueryType.REASONING: [
            r"\b(why|how does|explain|analyze|compare|evaluate|assess|trade.?off"
            r"|reason|logic|should i|best way|what if|difference between|pros and cons"
            r"|cause|effect|impact|implication|consequence|strategy|approach"
            r"|distributed|architecture|system design|scalability|performance"
            r"|limitation|constraint|theorem|principle|theory|concept|mechanism)\b",
        ],
        QueryType.CREATIVE: [
            r"\b(write|create|generate|story|poem|blog|email|essay|draft|compose"
            r"|imagine|describe|narrative|character|plot|scene|copy|advertisement"
            r"|product description|social media|caption|slogan|tagline)\b",
        ],
        QueryType.SIMPLE: [
            r"\b(what is|define|meaning|who is|when was|how many|list of|name of"
            r"|capital of|translate|convert|what are the|give me a list)\b",
        ],
    }

    def classify(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        scores = {qtype: 0 for qtype in QueryType}

        for query_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, prompt_lower, re.IGNORECASE)
                scores[query_type] += len(matches)

        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return QueryType.UNKNOWN

        return best.value