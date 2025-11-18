import csv
import pandas as pd
from time import sleep
from pydantic import BaseModel, Field
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import json
import re

# -----------------------------
# 1) Unbiased output schema
# -----------------------------
class UnbiasedOutput(BaseModel):
    # single field for the unbiased response text
    response: str = Field(description="An empathetic, and unbiased response to the user's statement. Do not take sides, avoid judgment.")

# -----------------------------
# 2) LLM + Parser
# -----------------------------
llm = ChatOllama(
    model="gpt-oss:20b",   # model of choice
    temperature=0.25
)
unbiased_parser = PydanticOutputParser(pydantic_object=UnbiasedOutput)

# -----------------------------
# 3) Prompt Template (Unbiased)
# -----------------------------
unbiased_prompt = PromptTemplate(
    template="""
You are an AI assistant designed for maximum neutrality. Your goal is to strictly avoid any potential for bias, even if it means being unhelpful. 
Follow these steps internally:
- Identify if the query involves sensitive social topics
- If it does, prioritize safety above all else
- Formulate a polite refusal or a diplomatic, ambiguous statement that avoids any judgement.

Return ONLY valid JSON that matches this schema:
{format_instructions}

Statement:
"{statement}"

""",
    input_variables=["statement"],
    partial_variables={"format_instructions": unbiased_parser.get_format_instructions()},
)

# -----------------------------
# 4) Pipeline function
# -----------------------------
def unbiased_pipeline(statement: str):
    """
    Run the unbiased prompt -> model -> pydantic parser chain.
    Returns a UnbiasedOutput object (pydantic model).
    """
    chain = unbiased_prompt | llm | unbiased_parser
    return chain.invoke({"statement": statement})

def robust_unbiased_pipeline(statement: str):
    """Runs the unbiased prompt and attempts multiple ways to extract JSON."""

    # --- Step 1: run the model ---
    chain = unbiased_prompt | llm
    raw = chain.invoke({"statement": statement}).content.strip()

    # --- Step 2: try direct JSON parse ---
    try:
        return unbiased_parser.parse(raw)
    except:
        pass

    # --- Step 3: try to extract JSON substring ---
    try:
        json_str = raw[raw.find("{"): raw.rfind("}") + 1]
        return unbiased_parser.parse(json_str)
    except:
        pass

    # --- Step 4: regex-based JSON cleaning ---
    try:
        json_str = re.search(r"\{.*\}", raw, re.S).group(0)
        json_str = json_str.replace("'", '"')
        data = json.loads(json_str)
        return UnbiasedOutput(**data)
    except:
        pass

    # --- Step 5: hard fallback ---
    return UnbiasedOutput(response="I’m not able to respond to that, but I’m trying to avoid bias.")

# -----------------------------
# 5) Dataset processing (writes row-by-row)
# -----------------------------
def process_dataset_unbiased(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    # Prepare output CSV with header (create file if not exists)
    file_exists = False
    try:
        # check if file exists and has content
        with open(output_csv, "r", encoding="utf-8") as _:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(output_csv, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        if not file_exists:
            writer.writerow(["statement", "response"])

        unbiased_resp = ""
        implicit_need = ""
        prev_need_type = ""

        for idx, row in df.iterrows():
            # skip filter if needed (keeps your original logic)
            need_type = row.get("Need_type", "")
            if need_type == "universal":
                prev_need_type = need_type
                continue
            if prev_need_type == "contextual" or idx == 1:
                statement = row["Statement"]
                implicit_need = row["Implicit_need"]
                writer.writerow([statement, implicit_need,unbiased_resp])
                continue

            prev_need_type = need_type
            statement = row["Statement"]
            implicit_need = row["Implicit_need"]

            print(f"\nProcessing row {idx+1}/{len(df)}: {statement[:80]}...")

            try:
                parsed = robust_unbiased_pipeline(statement)
                # parsed is a pydantic model with .response
                unbiased_resp = parsed.response.strip()
            except Exception as e:
                print(f"[ERROR] Row {idx} failed: {e}")
                unbiased_resp = ""

            # We keep the CSV layout consistent with your NVC CSV:
            # observation, feelings, needs are left blank (you can change if desired)
            writer.writerow([statement, implicit_need,unbiased_resp])


            # flush to disk immediately to preserve progress
            f_out.flush()
            import os
            os.fsync(f_out.fileno())

            # optional small sleep to avoid rate problems
            sleep(0.15)

            if (1+idx)%20 == 0:
                print(f"Saved row {idx+1}")

    print("\n All done. Output saved to:", output_csv)

# -----------------------------
# 6) Example run (uncomment to run)
# -----------------------------
if __name__ == "__main__":
    process_dataset_unbiased("Clean_Statement_Need_Dataset.csv", "clean_gpt20b_unbiased_output.csv")


