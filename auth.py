import streamlit_authenticator as stauth
import streamlit as st

def load_authenticator():
    config = {
        "credentials": {
            "usernames": {
                key: dict(st.secrets[f"credentials.usernames.{key}"])
                for key in st.secrets["credentials.usernames"].keys()
            }
        },
        "cookie": dict(st.secrets["cookie"])
    }
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"]
    )
