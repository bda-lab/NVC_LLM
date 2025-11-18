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
# 1) vanilla output schema
# -----------------------------
class VanillaOutput(BaseModel):
    # single field for the vanilla response text
    response: str = Field(description="Respond to the user as an empathetic AI assistant.")

# -----------------------------
# 2) LLM + Parser
# -----------------------------
llm = ChatOllama(
    model="gpt-oss:20b",  #llama3.1:8b
    temperature=0.25
)
vanilla_parser = PydanticOutputParser(pydantic_object=VanillaOutput)

# -----------------------------
# 3) Prompt Template (vanilla\)
# -----------------------------
vanilla_prompt = PromptTemplate(
    template="""
You are an empathetic AI assistant. Respond to the user's statement.

Return ONLY valid JSON that matches this schema:
{format_instructions}

Statement:
"{statement}"

""",
    input_variables=["statement"],
    partial_variables={"format_instructions": vanilla_parser.get_format_instructions()},
)

# -----------------------------
# 4) Pipeline function
# -----------------------------
def vanilla_pipeline(statement: str):
    """
    Run the vanilla prompt -> model -> pydantic parser chain.
    Returns a VanillaOutput object (pydantic model).
    """
    chain = vanilla_prompt | llm | vanilla_parser
    return chain.invoke({"statement": statement})

def robust_vanilla_pipeline(statement: str):
    """Runs the vanilla prompt and attempts multiple ways to extract JSON."""

    # --- Step 1: run the model ---
    chain = vanilla_prompt | llm
    raw = chain.invoke({"statement": statement}).content.strip()
    print("raw="+raw)

    # --- Step 2: try direct JSON parse ---
    try:
        van = vanilla_parser.parse(raw)
        print("van = "+van)
        return van
    except:
        pass

    # --- Step 3: try to extract JSON substring ---
    try:
        json_str = raw[raw.find("{"): raw.rfind("}") + 1]
        return vanilla_parser.parse(json_str)
    except:
        pass

    # --- Step 4: regex-based JSON cleaning ---
    try:
        json_str = re.search(r"\{.*\}", raw, re.S).group(0)
        json_str = json_str.replace("'", '"')
        data = json.loads(json_str)
        return VanillaOutput(**data)
    except:
        pass

    # --- Step 5: hard fallback ---
    return VanillaOutput(response="I’m not able to respond to that.")

# -----------------------------
# 5) Dataset processing (writes row-by-row)
# -----------------------------
def process_dataset_vanilla(input_csv: str, output_csv: str):
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

        vanilla_resp = ""
        implicit_need = ""
        prev_need_type = ""

        for idx, row in df.iterrows():
            # skip filter if needed (keeps your original logic)
            need_type = row.get("Need_type", "")
            if need_type == "universal":
                prev_need_type = need_type
                continue
            if prev_need_type == "contextual":
                statement = row["Statement"]
                implicit_need = row["Implicit_need"]
                writer.writerow([statement, implicit_need,vanilla_resp])
                continue
            
            prev_need_type = need_type

            statement = row["Statement"]
            implicit_need = row["Implicit_need"]

            print(f"\nProcessing row {idx+1}/{len(df)}: {statement[:80]}...")

            try:
                parsed = vanilla_pipeline(statement)
                # parsed is a pydantic model with .response
                vanilla_resp = parsed.response.strip()
            except Exception as e:
                print(f"[ERROR] Row {idx} failed: {e}")
                vanilla_resp = ""

            # We keep the CSV layout consistent with your NVC CSV:
            # observation, feelings, needs are left blank (you can change if desired)
            writer.writerow([statement, implicit_need,vanilla_resp])


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
# 6) Run
# -----------------------------
if __name__ == "__main__":
    process_dataset_vanilla("Clean_Statement_Need_Dataset.csv", "clean_gpt20_vanilla_output.csv")


