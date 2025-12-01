from qiskit_ibm_provider import IBMProvider

# Your API Key is the 'token'
my_token = "lXjps48H18MO5VY-I6ViAcFkYM_Li0SB1Vwbg7UBUYBk"

try:
    # This saves the key securely to a file on your computer
    IBMProvider.save_account(token=my_token, overwrite=True)
    print("Success! Your IBM Quantum token has been saved.")
except Exception as e:
    print(f"Error saving account: {e}")
    