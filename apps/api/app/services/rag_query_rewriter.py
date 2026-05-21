import re


class RuleBasedMultiQueryRewriter:
    """Safe local multi-query rewriting for technical GitHub issue search.

    This avoids LLM/API cost for now and preserves exact code-shaped terms.
    """

    def rewrite(self, question: str) -> list[str]:
        cleaned_question = self._clean(question)
        technical_terms = self._extract_technical_terms(cleaned_question)

        queries: list[str] = [cleaned_question]

        if technical_terms:
            joined_terms = " ".join(technical_terms)

            queries.append(
                f"{joined_terms} resolved issue fix maintainer answer"
            )
            queries.append(
                f"{joined_terms} bug reproduction expected behavior observed behavior"
            )
        else:
            queries.append(
                f"{cleaned_question} resolved issue maintainer answer"
            )
            queries.append(
                f"{cleaned_question} fix discussion reproduction"
            )

        return self._unique_preserve_order(queries)

    def _clean(self, text: str) -> str:
        return " ".join(text.split()).strip()

    def _extract_technical_terms(self, text: str) -> list[str]:
        patterns = [
            # Error constants: ERR_BUFFER_OUT_OF_BOUNDS
            r"\b[A-Z][A-Z0-9_]{3,}\b",

            # JS errors: RangeError, TypeError, SyntaxError
            r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b",

            # Dotted APIs: Buffer.toString, fs.readFile
            r"\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+\b",

            # Function calls: toString(), iterate()
            r"\b[A-Za-z_$][\w$]*\(\)",

            # Versions: v22.7.0, 22.7.0
            r"\bv?\d+\.\d+(?:\.\d+)?\b",

            # Paths / node internals: node:internal/buffer, lib/buffer.js
            r"\b[A-Za-z0-9_@.-]+[:/][A-Za-z0-9_@./:-]+\b",
        ]

        terms: list[str] = []

        for pattern in patterns:
            terms.extend(re.findall(pattern, text))

        return self._remove_generic_terms(
            self._unique_preserve_order(terms)
        )

    def _remove_generic_terms(self, terms: list[str]) -> list[str]:
        generic_terms = {
            "node",
            "node.js",
            "nodejs",
            "javascript",
            "js",
            "issue",
            "bug",
            "error",
        }

        return [
            term
            for term in terms
            if term.lower().strip() not in generic_terms
        ]

    def _unique_preserve_order(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []

        for value in values:
            key = value.lower()

            if key in seen:
                continue

            seen.add(key)
            unique.append(value)

        return unique