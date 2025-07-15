from typing import Union
from typing import TypedDict
class Person(TypedDict):
	name: str
	age: int
	is_student: bool
person: Person = {
	"name":"Alice",
	"age":30,
	"is_student":False
}
def process(value: Union[int,str])-> str:
	if isinstance(value, int):
		return f"Number doubled: {value*2}"
	else:
		return f"Text uppercased: {value.upper()}"
print(person["name"])
print(person["age"])
print(process(4))
print(process("Hello"))
