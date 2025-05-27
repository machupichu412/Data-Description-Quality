from ollama import Client
from ollama import generate

client = Client()
model_resp = client.create(
  model='test_tool_1',
  from_='vicuna',
  system="""You are an AI text quality reviewer tool. Your task is to review technical descriptions for data entities and attributes. Based on the quality of the description, you will output either 'Pass' or 'Fail'.
'Pass' means the description meets the quality standards and provides clear, accurate, and complete information.
'Fail' means the description does not meet the required standards, and you must provide a specific reason for the failure.
Your output format will be: <Pass or Fail>, <N/A or reason for failure>
Examples:
Input: 'Indirect mapping to 'LicensingProgram' column from 'hub_vw_licensingprogram'
Output: 'Fail, Missing on how to join conditions and table information needs to be specified'

Input: 'This Attribute is generated based on 'SavedDays' column coming from the WorkFactSRReactiveRetention from UDP MC:
Logic: when 'SavedDays' column is null hardcoded as 'Unspecified'
when 'SavedDays' column is between 0 to 30 then hardcoded as '<31'
when 'SavedDays' column is between 31 to 60 then hardcoded as '31'
when 'SavedDays' column is between 61 to 90 then hardcoded as '61'
when 'SavedDays' column is between 91 to 120 then hardcoded as '91'
when 'SavedDays' column is between 121 to 179 then hardcoded as '121'
when 'SavedDays' column is between 180 to 364 then hardcoded as '180'
otherwise hardcoded as '365''
Output: 'Pass, N/A'""",
  stream=False,
)
print(model_resp.status)

response = generate('test_tool_1', 'Direct Mapping to CaseTransferReason column in Cases table coming from DFM Events data processed by Cornerstone having String values')
print(response['response'])

response = generate('test_tool_1', 'Surrogate key generated within UDP NRT to identify unique records of FactSREvents table.')
print(response['response'])