import os
import re
import json
import docx
from datetime import datetime

class ScheduleParser(object):
    def __init__(self, save_dir = None):
        self.save_dir = self.check_save_dir(save_dir)
        self.return_dir = os.path.join(self.save_dir, "WeeklySchedule.json")
        self.WeeklySchedule = self.load_stored_events() or []
        self.supported_formats = [".docx"]
        
    def check_save_dir(self, save_dir):
        save_dir = save_dir or os.path.join(os.getcwd(), "server", "app", "data", "WeeklySchedule")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        return save_dir
    
    def save_stored_events(self):
        with open(self.return_dir, "w", encoding = "utf-8") as file:
            json.dump(self.WeeklySchedule, file, indent = 4) 
        
    def load_stored_events(self):
        try:
            with open(self.return_dir, "r", encoding = "utf-8") as file:
                return json.load(file)
        except:
            return []

    def get_events(self):
        return self.WeeklySchedule
    
    def information_parser(self, text):
        time_pattern = re.compile(r"\b\d{2}\.\d{2}\b")
        tp_pattern = re.compile(r"TP: [^\n]*")
        dd_pattern = re.compile(r"DD: [^\n]*")
        cb_pattern = re.compile(r"C/b: [^\n]*")
        
        if re.search(time_pattern, text):
            return "time"

        if re.search(tp_pattern, text):
            return "attendees"
        
        if re.search(dd_pattern, text):
                return "location"
            
        if re.search(cb_pattern, text):
                return "preparation"
        return ""
    
    def events_parser(self, file_dir):
        date_pattern = re.compile(r"Thứ\s*[A-Za-zÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯẮẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưắạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỡợỤỦỨỪỬỮỰỲỴÝỶỸấầẩẫậắằẳẵặẹẻẽềểểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]+,\s*ngày\s*\d{1,2}/\d{1,2}")        
        events = []
        current_event = {}
        current_day: str = None
        document = docx.Document(file_dir)
        for table in document.tables:
            rows = table.rows
            for row in rows:
                first_cell = row.cells[0].text.strip()
                second_cell = row.cells[1].text.strip()
                
                if re.search(date_pattern, first_cell):
                    current_day = first_cell
            
                    continue
                
                for line in first_cell.split("\n"):
                    if self.information_parser(line) == "location":
                        current_event.update({"location": line.split(": ")[1]})
                        current_event.update({"date": current_day})
                        events.append(current_event)
                        current_event = {}
                        continue

                    if self.information_parser(line) == "time":
                        temp = line.split(": ")
                        current_event.update({"time": temp[0]})
                        current_event.update({"name": temp[1]})
                        continue
                
                    temp = line.split(": ")
                    if len(temp) == 2:
                        current_event.update({self.information_parser(line): temp[1]})
                        continue

                for line in second_cell.split("\n"):
                    if self.information_parser(line) == "location":
                        current_event.update({"location": line.split(": ")[1]})
                        current_event.update({"date": current_day})
                        events.append(current_event)
                        current_event = {}
                        continue

                    if self.information_parser(line) == "time":
                        temp = line.split(": ")
                        current_event.update({"time": temp[0]})
                        current_event.update({"name": temp[1]})
                        continue
                
                    temp = line.split(": ")
                    if len(temp) == 2:
                        current_event.update({self.information_parser(line): temp[1]})
                        continue
        return events
                     
    def format_events(self, file_dir):
        events = self.events_parser(file_dir)
        formatted_events = []
        current_year = datetime.now().year
        
        for event in events:
            formatted_event = event.copy()
            
            # Extract date
            date_match = re.search(r"Thứ [^,]+, ngày (\d{1,2})/(\d{1,2})", event['date'])
            if date_match:
                day, month = map(int, date_match.groups())
                
                # Extract time
                time_parts = event['time'].split('.')
                hour, minute = map(int, time_parts)
                
                # Create ISO datetime
                try:
                    dt = datetime(current_year, month, day, hour, minute)
                    formatted_event['iso_datetime'] = dt.isoformat()
                    
                    # Remove separate date and time fields if desired
                    del formatted_event['date']
                    del formatted_event['time']
                    
                except ValueError as e:
                    print(f"Error processing date/time for event: {e}")
                    formatted_event['iso_datetime'] = None
            
            formatted_events.append(formatted_event)
        
        self.WeeklySchedule = formatted_event
        self.save_stored_events()
