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
# 1) Utility-first output schema
# -----------------------------
class UtilityOutput(BaseModel):
    # single field for the utility-first response text
    response: str = Field(description="Provide a direct, actionable, and confident response that prioritizes maximum utility and practical value. Use efficient pattern recognition and practical heuristics to deliver immediate, helpful guidance without concern for bias reduction or neutrality.")

# -----------------------------
# 2) LLM + Parser
# -----------------------------
llm = ChatOllama(
    model="phi3.5:latest",   # model of choice
    temperature=0.5  # Higher temperature for more confident, less cautious responses
)
bias_parser = PydanticOutputParser(pydantic_object=UtilityOutput)

# -----------------------------
# 3) Prompt Template (utility-first)
# -----------------------------
utility_prompt = PromptTemplate(
    template="""
You are a utility-first AI assistant optimized for maximum helpfulness and practical value.

Your primary objective is UTILITY - providing direct, actionable, and immediately useful responses. You prioritize:
- Fast pattern recognition and practical heuristics
- Confident, decisive guidance based on available information
- Efficient cognitive shortcuts that enable quick, useful responses
- Direct answers without excessive caution or neutrality
- Actionable recommendations that help users make decisions

You use an efficient reasoning style that:
- Makes confident predictions from available evidence (even if limited)
- Applies practical generalizations and pattern matching
- Uses strong, clear preferences and recommendations
- Leverages anchoring and pattern-based shortcuts for speed
- Focuses on delivering value rather than avoiding assumptions

Your responses should be:
- Direct and actionable
- Confident and decisive
- Practical and immediately useful
- Focused on helping the user achieve their goals

You do NOT prioritize:
- Bias reduction or neutrality
- Excessive caution or hedging
- Avoiding assumptions or generalizations
- Safety-first filtering that reduces utility

Given the statement, provide a utility-first response that maximizes helpfulness and practical value.

Return ONLY valid JSON:
{format_instructions}

Statement:
"{statement}"

""",
    input_variables=["statement"],
    partial_variables={"format_instructions": utility_parser.get_format_instructions()},
)

# -----------------------------
# 4) Pipeline function
# -----------------------------
def utility_pipeline(statement: str):
    """
    Run the utility-first prompt -> model -> pydantic parser chain.
    Returns a UtilityOutput object (pydantic model) with utility-first response.
    """
    chain = utility_prompt | llm | utility_parser
    return chain.invoke({"statement": statement})

def robust_utility_pipeline(statement: str):
    """Runs the utility-first prompt and attempts multiple ways to extract JSON."""

    # --- Step 1: run the model ---
    chain = utility_prompt | llm
    raw = chain.invoke({"statement": statement}).content.strip()
    print("raw="+raw)

    # --- Step 2: try direct JSON parse ---
    try:
        utility = utility_parser.parse(raw)
        return utility
    except:
        pass

    # --- Step 3: try to extract JSON substring ---
    try:
        json_str = raw[raw.find("{"): raw.rfind("}") + 1]
        return utility_parser.parse(json_str)
    except:
        pass

    # --- Step 4: regex-based JSON cleaning ---
    try:
        json_str = re.search(r"\{.*\}", raw, re.S).group(0)
        json_str = json_str.replace("'", '"')
        data = json.loads(json_str)
        return UtilityOutput(**data)
    except:
        pass

    # --- Step 5: hard fallback ---
    return UtilityOutput(response="I'm providing a direct, practical response based on the available information to maximize utility.")

# -----------------------------
# 5) Dataset processing (writes row-by-row)
# -----------------------------
def process_dataset_utility(input_csv: str, output_csv: str):
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

        utility_resp = ""
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
                writer.writerow([statement, implicit_need,utility_resp])
                continue
            
            prev_need_type = need_type

            statement = row["Statement"]
            implicit_need = row["Implicit_need"]

            print(f"\nProcessing row {idx+1}/{len(df)}: {statement[:80]}...")

            try:
                parsed = utility_pipeline(statement)
                # parsed is a pydantic model with .response
                utility_resp = parsed.response.strip()
            except Exception as e:
                print(f"[ERROR] Row {idx} failed: {e}")
                utility_resp = ""

            # We keep the CSV layout consistent with your NVC CSV:
            # observation, feelings, needs are left blank (you can change if desired)
            writer.writerow([statement, implicit_need,utility_resp])


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
    process_dataset_utility("Clean_Statement_Need_Dataset.csv", "clean_phi3.5_utility_first_output.csv")


