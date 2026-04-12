# --- Tab 1-la irukura Prediction Button logic-a ippadi maathunga ---

if st.button("Predict Loan Eligibility"):
    # First, model kitta prediction vangunga
    prob = model.predict_proba(input_df)[0][1]
    chance = round(prob * 100, 2)
    
    # --- DYNAMIC OVERRIDE LOGIC (Fixing the Error) ---
    # Credit score romba kammiya irundha, namma model result-a force-ah maathiduvom
    if credit_score < 500:
        res_text = "REJECTED"
        chance = min(chance, 30.0) # Chance-a auto-ah 30%-kku kila kondu vandhiduvom
        st.error(f"⚠️ High Risk: Credit Score ({credit_score}) is too low for approval.")
    else:
        res_text = "APPROVED" if chance >= 50 else "REJECTED"
        
    # Session state update
    st.session_state['chance'] = chance
    
    # Final Result Display
    if res_text == "APPROVED":
        st.success(f"Final Decision: {res_text} ({chance}% Confidence)")
    else:
        st.error(f"Final Decision: {res_text} ({chance}% Confidence)")
