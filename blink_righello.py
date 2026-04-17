from machine import ADC, Pin
from utime import sleep_ms

pin = Pin("GP25", Pin.OUT)
sensore = ADC(Pin(26)) # LDR, strumento per misurare la luminosità

data = []

try:
    while True:
     valore = 65535 - sensore.read_u16() # legge il valore del sensore
     data.append(valore)
     sleep_ms(100)

except KeyboardInterrupt:
    with open("data.txt", "w") as f:
        for d in data:
            f.write(str(d) + "\n")
    print ("Data saved to data.txt")
    print ("media: ", sum(data)/len(data))
    print ("sigma: ", (sum([(x - sum(data)/len(data))**2 for x in data])/len(data))**0.5)