"""The following code is to create a database(a CSV file) which stores the name,age,contact number and
email ID of the student given as the input by the user

what does this program do??
1.Prompts the user for the user
2.Verifies the given input with the user asking if it is valid
3.If yes,writes the inpput as details into the CSV file using file handling operations
4.If not,asks the user to enter the correct details once again

library csv is used for csv file handling operations"""

import csv

#csv file handling operations
def write_into_csv(info_list):
    with open("student_info.csv","a",newline="") as csv_file:
        writer=csv.writer(csv_file)
        if csv_file.tell()==0:
            writer.writerow(["name","age","contact number","email ID"])
        writer.writerow(info_list)

# user input
if __name__=="__main__":
    condition=True
    student_num=1
    while(condition):
        student_info=input("student #{} information".format(student_num))
        # student_info=input("Enter the student details")

        student_info_list=student_info.split(" ")
        print("\nThe entered information-\nName:{}\nAge:{}\nContact Number:{}\nE-mail ID:{}"
              .format(student_info_list[0],student_info_list[1],student_info_list[2],student_info_list[3]))
        choice_check=input("Is the information given is valid? (yes/no)")
        if choice_check=="yes":
            write_into_csv(student_info_list)

            condition_check=input("Do you eant to enter another student's information? (yes/no)")
            if condition_check=="yes":
                condition=True
                student_num+=1
            elif condition_check=="no":
                condition=False
        elif choice_check=="no":
            print("\nPlease re-enter the values")

