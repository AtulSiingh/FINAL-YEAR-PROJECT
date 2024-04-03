hedule"):
        if not from_station_code or not to_station_code or not date_of_journey:
            st.warning("Please enter your Details care fully")
        else:
            train_schedule = get_train_schedule(from_station_code, to_station_code, date_of_journey)
        if train_schedule:
            data_list = train_schedule["data"]
            df4 = pd.DataFrame(data_list)
            table = PrettyTable()
            table.field_names = ["Train Number", "Train Name", "Run Days", "Source", "Destination", "Start Time", "Reach Time"]
            for data in data_list:
                run_days = ", ".join(data["run_days"])
                table.add_row([data["train_number"], data["train_name"], run_days, data["train_src"], data["train_dstn"],
                               data["from_std"], data["to_std"]])
            st.write(table)