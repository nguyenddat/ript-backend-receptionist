import codecs
tagIndex = {
        "020101": "Identity Code",
        "020102": "Name",
        "020103": "DOB",
        "020104": "Gender",
        "020105": "Nationality",
        "020106": "Ethnic",
        "020107": "Religion",
        "020108": "Hometown",
        "020109": "Permanent Address",
        "02010A": "Identifying Features",
        "02010B": "Card Issuance Date",
        "02010C": "Expiration Date",
        "02010D": "Parents' Names"
    }

def extract_data(data):

    so_cccd = ""
    ten = ""
    data_extract = {
        "Identity Code" : "",
        "Name" : "",
        "DOB" : "",
        "Gender" : "",
        "Nationality" : "",
        "Ethnic" : "",
        "Religion" : "",
        "Hometown" : "",
        "Permanent Address" : "",
        "Identifying Features" : "",
        "Card Issuance Date" : "",
        "Expiration Date" : "",
    }



    data_all = ""
    for i in data:
        data_all = data_all + " " +  i
    # print(data_all)

    # hexBytes = bytes.fromhex(data_all)

    # print(hexBytes)

    offset = 0

    # while (offset < 2000):
    #     if (data_all):
    #         break
    #     offset +=1
    while (offset < 2000):

        # GET CCCD
        try:
            if not data_extract.get("Identity Code") and str(data[offset]) == "30" and  str(data[offset + 2]) == "02" and  str(data[offset + 3]) == "01" and  str(data[offset + 4]) == "01":
                data_tmp = ""
                for i in range(int(data[offset + 6], 16)):
                    data_tmp = data_tmp +  data[offset + 7 + i]
                data_tmp = str(codecs.decode(data_tmp, 'hex').decode('utf-8'))
                data_extract["Identity Code"] = data_tmp
        except:
            pass
        # GET Tên
        try:
            if not data_extract.get("Name") and str(data[offset]) == "30" and  str(data[offset + 2]) == "02" and  str(data[offset + 3]) == "01" and  str(data[offset + 4]) == "02":
                data_tmp = ""
                for i in range(int(data[offset + 6], 16)):
                    data_tmp = data_tmp +  data[offset + 7 + i]
                data_tmp = str(codecs.decode(data_tmp, 'hex').decode('utf-8'))
                data_extract["Name"] = data_tmp
        except:
            pass
        # GET DOB
        try:
            if not data_extract.get("DOB") and str(data[offset]) == "30" and  str(data[offset + 2]) == "02" and  str(data[offset + 3]) == "01" and  str(data[offset + 4]) == "03":
                data_tmp = ""
                for i in range(int(data[offset + 6], 16)):
                    data_tmp = data_tmp +  data[offset + 7 + i]
                data_tmp = str(codecs.decode(data_tmp, 'hex').decode('utf-8'))
                data_extract["DOB"] = data_tmp
        except:
            pass

        # GET Gender
        try:
            if not data_extract.get("Gender") and str(data[offset]) == "30" and  str(data[offset + 2]) == "02" and  str(data[offset + 3]) == "01" and  str(data[offset + 4]) == "04":
                data_tmp = ""
                for i in range(int(data[offset + 6], 16)):
                    data_tmp = data_tmp +  data[offset + 7 + i]
                data_tmp = str(codecs.decode(data_tmp, 'hex').decode('utf-8'))
                data_extract["Gender"] = data_tmp
        except:
            pass

        # GET Nationality
        try:
            if not data_extract.get("Nationality") and str(data[offset]) == "30" and  str(data[offset + 2]) == "02" and  str(data[offset + 3]) == "01" and  str(data[offset + 4]) == "05":
                data_tmp = ""
                for i in range(int(data[offset + 6], 16)):
                    data_tmp = data_tmp +  data[offset + 7 + i]
                data_tmp = str(codecs.decode(data_tmp, 'hex').decode('utf-8'))
                data_extract["Nationality"] = data_tmp
        except:
            pass

        # GET Ethnic
        try:
            if not data_extract.get("Ethnic") and str(data[offset]) == "30" and  str(data[offset + 2]) == "02" and  str(data[offset + 3]) == "01" and  str(data[offset + 4]) == "06":
                data_tmp = ""
                for i in range(int(data[offset + 6], 16)):
                    data_tmp = data_tmp +  data[offset + 7 + i]
                data_tmp = str(codecs.decode(data_tmp, 'hex').decode('utf-8'))
                data_extract["Ethnic"] = data_tmp
        except:
            pass

        # GET Religion
        try:
            if not data_extract.get("Religion") and str(data[offset]) == "30" and  str(data[offset + 2]) == "02" and  str(data[offset + 3]) == "01" and  str(data[offset + 4]) == "07":
                data_tmp = ""
                for i in range(int(data[offset + 6], 16)):
                    data_tmp = data_tmp +  data[offset + 7 + i]
                data_tmp = str(codecs.decode(data_tmp, 'hex').decode('utf-8'))
                data_extract["Religion"] = data_tmp
        except:
            pass

        # GET Hometown
        try:
            if not data_extract.get("Hometown") and str(data[offset]) == "30" and  str(data[offset + 2]) == "02" and  str(data[offset + 3]) == "01" and  str(data[offset + 4]) == "08":
                data_tmp = ""
                for i in range(int(data[offset + 6], 16)):
                    data_tmp = data_tmp +  data[offset + 7 + i]
                data_tmp = str(codecs.decode(data_tmp, 'hex').decode('utf-8'))
                data_extract["Hometown"] = data_tmp
        except:
            pass

        # GET Permanent Address
        try:
            if not data_extract.get("Permanent Address") and str(data[offset]) == "30" and  str(data[offset + 2]) == "02" and  str(data[offset + 3]) == "01" and  str(data[offset + 4]) == "09":
                data_tmp = ""
                for i in range(int(data[offset + 6], 16)):
                    data_tmp = data_tmp +  data[offset + 7 + i]
                data_tmp = str(codecs.decode(data_tmp, 'hex').decode('utf-8'))
                data_extract["Permanent Address"] = data_tmp
        except:
            pass

        # GET Identifying Features
        try:
            if not data_extract.get("Identifying Features") and str(data[offset]) == "30" and  str(data[offset + 2]) == "02" and  str(data[offset + 3]) == "01" and  str(data[offset + 4]) == "0A":
                data_tmp = ""
                for i in range(int(data[offset + 6], 16)):
                    data_tmp = data_tmp +  data[offset + 7 + i]
                data_tmp = str(codecs.decode(data_tmp, 'hex').decode('utf-8'))
                data_extract["Identifying Features"] = data_tmp
        except:
            pass


        # GET Card Issuance Date
        try:
            if not data_extract.get("Card Issuance Date") and str(data[offset]) == "30" and  str(data[offset + 2]) == "02" and  str(data[offset + 3]) == "01" and  str(data[offset + 4]) == "0B":
                data_tmp = ""
                for i in range(int(data[offset + 6], 16)):
                    data_tmp = data_tmp +  data[offset + 7 + i]
                data_tmp = str(codecs.decode(data_tmp, 'hex').decode('utf-8'))
                data_extract["Card Issuance Date"] = data_tmp
        except:
            pass

        # GET Card Expiration Date
        try:
            if not data_extract.get("Expiration Date") and str(data[offset]) == "30" and  str(data[offset + 2]) == "02" and  str(data[offset + 3]) == "01" and  str(data[offset + 4]) == "0C":
                data_tmp = ""
                for i in range(int(data[offset + 6], 16)):
                    data_tmp = data_tmp +  data[offset + 7 + i]
                data_tmp = str(codecs.decode(data_tmp, 'hex').decode('utf-8'))
                data_extract["Expiration Date"] = data_tmp
        except:
            pass
        
        offset +=1
    # result_string = hexBytes.decode("utf-8")
    return data_extract