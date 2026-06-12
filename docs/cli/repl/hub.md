# REPL: Package Hub Integration

Search and install versioned Logic Packs and models directly from the central registry.

---

## `.hub`
Performs hub registry interactions (search, install, and list local packages).

* **Syntax**: 
  * `.hub search <query>`
  * `.hub install <package_name>`
  * `.hub list`

---

## Examples

### Search
```sql
anam> .hub search fraud
financial-compliance@1.0.0 — Datalog rules and models for risk checking
```

### Install
```sql
anam> .hub install financial-compliance
Installing financial-compliance...
✓ Package financial-compliance@1.0.0 installed to ./packs/
```

### List
```sql
anam> .hub list
financial-compliance@1.0.0
```
