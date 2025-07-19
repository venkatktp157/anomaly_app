def load_authenticator():
    import streamlit_authenticator as stauth
    import streamlit as st

    # Dynamically extract all keys starting with "credentials.usernames."
    user_keys = [
        key.split(".")[-1]
        for key in st.secrets.keys()
        if key.startswith("credentials.usernames.")
    ]

    usernames = {
        uname: dict(st.secrets[f"credentials.usernames.{uname}"])
        for uname in user_keys
    }

    cookie = dict(st.secrets["cookie"])

    st.write("Available keys:", list(st.secrets.keys()))

    return stauth.Authenticate(
        {"usernames": usernames},
        cookie["name"],
        cookie["key"],
        cookie["expiry_days"]
    )
    