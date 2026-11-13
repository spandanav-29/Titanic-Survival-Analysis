import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Titanic Survival Analysis", page_icon="🚢", layout="wide")

st.title("🚢 Titanic Survival Analysis")
st.write("A data analysis project using PostgreSQL, Python, Pandas and Streamlit.")

@st.cache_data
def load_data():
    conn = psycopg2.connect(**st.secrets["postgres"])
    df = pd.read_sql_query(
        "SELECT * FROM titanic_project.passengers ORDER BY passengerid",
        conn
    )
    conn.close()
    return df

try:
    df = load_data()

    total = len(df)
    survivors = int(df["survived"].sum())
    non_survivors = total - survivors
    rate = survivors / total * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Passengers", total)
    c2.metric("Survivors", survivors)
    c3.metric("Non-Survivors", non_survivors)
    c4.metric("Survival Rate", f"{rate:.2f}%")

    st.subheader("Passenger Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Survival by Gender")
    gender = df.groupby("sex", as_index=False).agg(
        total_passengers=("survived", "count"),
        survivors=("survived", "sum")
    )
    gender["survival_rate"] = gender["survivors"] / gender["total_passengers"] * 100
    st.plotly_chart(px.bar(gender, x="sex", y="survival_rate",
                           color="sex", title="Survival Rate by Gender",
                           labels={"survival_rate": "Survival Rate (%)"}),
                    use_container_width=True)

    st.subheader("Survival by Passenger Class")
    class_data = df.groupby("pclass", as_index=False).agg(
        survival_rate=("survived", "mean")
    )
    class_data["survival_rate"] *= 100
    st.plotly_chart(px.bar(class_data, x="pclass", y="survival_rate",
                           title="Survival Rate by Passenger Class",
                           labels={"pclass": "Passenger Class",
                                   "survival_rate": "Survival Rate (%)"}),
                    use_container_width=True)

    st.subheader("Survival Distribution")
    distribution = pd.DataFrame({
        "Status": ["Survived", "Did Not Survive"],
        "Count": [survivors, non_survivors]
    })
    st.plotly_chart(px.pie(distribution, names="Status", values="Count",
                           title="Passenger Survival Distribution"),
                    use_container_width=True)

    st.subheader("Filter Passengers")
    selected_sex = st.multiselect("Select gender", df["sex"].dropna().unique(),
                                  default=list(df["sex"].dropna().unique()))
    selected_class = st.multiselect("Select passenger class",
                                    sorted(df["pclass"].dropna().unique()),
                                    default=sorted(df["pclass"].dropna().unique()))
    filtered = df[df["sex"].isin(selected_sex) & df["pclass"].isin(selected_class)]
    st.dataframe(filtered, use_container_width=True)
    st.success("Dashboard loaded successfully from PostgreSQL!")

except Exception as e:
    st.error(f"Database connection error: {e}")
