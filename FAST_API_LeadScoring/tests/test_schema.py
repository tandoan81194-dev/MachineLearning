import unittest

from pydantic import ValidationError

from app.schemas import AnalysisRunInput, LeadInput


class LeadInputSchemaTest(unittest.TestCase):
    def test_lead_input_accepts_valid_payload(self) -> None:
        payload = {
            "city": "city_103",
            "gender": "Male",
            "enrolled_university": "no_enrollment",
            "education_level": "Graduate",
            "major_discipline": "STEM",
            "relevent_experience": "Has relevent experience",
            "experience": ">20",
            "company_size": "50-99",
            "company_type": "Pvt Ltd",
            "last_new_job": "1",
            "training_hours": 36,
            "city_development_index": 0.92,
        }

        lead = LeadInput(**payload)

        self.assertEqual(lead.city, "city_103")
        self.assertEqual(lead.training_hours, 36)

    def test_lead_input_rejects_unexpected_fields(self) -> None:
        payload = {
            "city": "city_103",
            "training_hours": 36,
            "unexpected": "value",
        }

        with self.assertRaises(ValidationError) as context:
            LeadInput(**payload)

        self.assertIn("unexpected", str(context.exception))

    def test_analysis_run_accepts_employee_metadata_and_lead_payload(self) -> None:
        payload = {
            "source_filename": "employees.xlsx",
            "source_sheet": "Unseen Employees",
            "employees": [
                {
                    "employee_id": "EMP-0001",
                    "employee_name": "Nguyen Minh Anh",
                    "department": "Data",
                    "role": "Senior Data Analyst",
                    "manager": "Tran Quang Huy",
                    "scenario_note": "Excel upload row",
                    "source_row": 5,
                    "lead": {
                        "city": "city_103",
                        "training_hours": 36,
                    },
                }
            ],
        }

        analysis = AnalysisRunInput(**payload)

        self.assertEqual(analysis.source_filename, "employees.xlsx")
        self.assertEqual(analysis.employees[0].employee_id, "EMP-0001")
        self.assertEqual(analysis.employees[0].lead.city, "city_103")


if __name__ == "__main__":
    unittest.main()
