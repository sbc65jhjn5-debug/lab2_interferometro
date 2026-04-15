from machine import ADC, Pin
from utime import sleep_ms

pin = Pin("GP25", Pin.OUT)
sensore = ADC(Pin(26)) # LDR, strumento per misurare la luminosità

data = []

try:
    while True:
        valore = 65535 - sensore.read_u16() # legge il valore del sensore
        data.append(valore)
        print(valore)
        sleep_ms(10) # "dorme" per 10 ms

except KeyboardInterrupt:
    with open("data.txt", "w") as f:
        for d in data:
            f.write(str(d) + "\n")
    print ("Data saved to data.txt")
