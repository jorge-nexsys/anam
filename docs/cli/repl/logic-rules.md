# REPL: logic & Rules Management

Register and inspect compiled symbolic Datalog reasoning rules and logic guardrails.

---

## `.logic`
Compiles and registers a symbolic Datalog rule/constraint into the logical optimization layer.

* **Syntax**: `.logic <rule_name> "<datalog_expression>"`
* **Example**:
  ```sql
  anam> .logic high_risk "fraud_prob > 0.90 AND amount > 10000"
  ✓ Registered rule 'high_risk'
  ```

---

## `.nl`
Translates a natural language constraint description into a compiled Datalog rule via an LLM.

* **Syntax**: `.nl <rule_name> <table_name> <english_description>`
* **Requirements**: Requires a valid API Key in the `.env` file (`ANAM_LLM_API_KEY` or `OPENAI_API_KEY`).
* **Example**:
  ```sql
  anam> .nl night_risk txns Flag any transaction between midnight and 5am over $5000
  Compiling NL → Datalog via LLM...
  ✓ Generated and registered rule 'night_risk':
    Datalog: night_risk(X) :- txns(X), X.time >= '00:00', X.time < '05:00', X.amount > 5000.
  ```

---

## `.rules`
Lists all compiled and currently active Datalog constraints in the query optimizer context.

* **Example**:
  ```sql
  anam> .rules
  +-----------+----------------------------------------+
  | Name      | Datalog Source                         |
  +-----------+----------------------------------------+
  | high_risk | fraud_prob > 0.90 AND amount > 10000   |
  +-----------+----------------------------------------+
  ```
