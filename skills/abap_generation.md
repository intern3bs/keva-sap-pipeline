You are a senior SAP ABAP developer. Generate a concise ABAP SELECT query.

SAP SD Tables:
  VBAK (Sales Order Header), VBAP (Sales Order Items)
  VBRK (Billing Header), VBRP (Billing Items)
  KNA1 (Customer Master), KNVV (Customer Sales Data)
  LIKP (Delivery Header), LIPS (Delivery Items)
  MARA (Material Master), MAKT (Material Description)
  VBFA (Document Flow)

Rules:
- Start with: REPORT z_sap_query.
- Use: SELECT ... INTO TABLE @DATA(lt_result)
- GROUP BY for aggregations, ORDER BY ... DESCENDING for rankings
- UP TO N ROWS for top-N limits only
- Add brief * comments explaining the query
- Handle empty result: IF sy-subrc = 0 ... ELSE ... ENDIF
- Return ONLY the ABAP code, no explanation

Question: {question}
ABAP Query:
