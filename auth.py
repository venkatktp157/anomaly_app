import streamlit as st
import streamlit_authenticator as stauth
import bcrypt

def load_authenticator():
    # 🔍 Step 1: Extract usernames from secrets
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

    # 🔐 Step 2: Create authenticator instance
    authenticator = stauth.Authenticate(
        {"usernames": usernames},
        cookie["name"],
        cookie["key"],
        cookie["expiry_days"]
    )

    # 🧪 Optional Debug Mode (dev only)
    if st.sidebar.checkbox("Enable Auth Debug Mode"):
        st.write("Available usernames:", list(usernames.keys()))

        selected_user = st.selectbox("Select user to test", usernames.keys())
        test_password = st.text_input("Enter raw password to verify", type="password")

        if test_password:
            hashed_pw = usernames[selected_user]["password"]
            match = bcrypt.checkpw(test_password.encode(), hashed_pw.encode())

            st.success("✅ Password matches!" if match else "❌ Incorrect password for user")

    return authenticator
