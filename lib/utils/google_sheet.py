"""
This is a wrapper class for Google sheet
"""
from typing import List

import gspread

from lib.config import credentials


class GoogleSheet:
    def __init__(self, sheet_name: str, worksheet_name: str, duplicate_from: str = None):
        gc = gspread.service_account_from_dict(credentials)
        self.benchmark_sheet = gc.open(sheet_name)
        worksheet_titles = [w.title for w in self.benchmark_sheet.worksheets()]

        self.worksheet_name = worksheet_name
        if worksheet_name not in worksheet_titles:
            if duplicate_from is not None:
                self.worksheet = self.benchmark_sheet.duplicate_sheet(self.benchmark_sheet.worksheet(duplicate_from).id,
                                                                      new_sheet_name=worksheet_name)
                self.get_headers()
            else:
                self.worksheet = self.benchmark_sheet.add_worksheet(worksheet_name, 100, 100)
                self.headers = []
                self.headers_to_idx = {}
        else:
            self.worksheet = self.benchmark_sheet.worksheet(worksheet_name)
            self.get_headers()

    def set_headers(self, headers: List[str]):
        self.headers = headers
        self.headers_to_idx = {h: i for i, h in enumerate(self.headers)}
        self.worksheet.update('A1', [self.headers])

    def get_headers(self):
        self.headers = self.worksheet.row_values(1)
        self.headers_to_idx = {h: i for i, h in enumerate(self.headers)}

    def get_row(self, row_idx: int):
        return self.worksheet.row_values(row_idx)

    def get_col(self, col_idx: int):
        return self.worksheet.col_values(col_idx)

    def get_col_by_header(self, header: str):
        return self.get_col(self.get_idx(header) + 1)

    def get_idx(self, header: str):
        print(self.headers_to_idx)
        return self.headers_to_idx[header]

    def get_cell(self, row_idx: int, col_idx: int):
        return self.worksheet.cell(row_idx, col_idx).value

    def get_cell_by_header(self, row_idx: int, header: str):
        return self.get_cell(row_idx, self.get_idx(header))

    def set_cell(self, row_idx: int, col_idx: int, value):
        self.worksheet.update_cell(row_idx, col_idx, value)

    def set_cell_by_header(self, row_idx: int, header: str, value):
        self.set_cell(row_idx, self.get_idx(header), value)

    def get_all_values(self) -> List[List[str]]:
        return self.worksheet.get_all_values()

    def get_all_records(self) -> List[dict]:
        return self.worksheet.get_all_records()

    def bulk_update(self, records: List[dict]):

        # Create a list of lists representing the rows to insert into the sheet
        rows_to_insert = [list(self.headers)]
        for row in records:
            new_row = []
            for header in self.headers:
                if header in row:
                    new_row.append(row[header])
                else:
                    new_row.append('')
            rows_to_insert.append(new_row)

        self.worksheet.update('A1', rows_to_insert)