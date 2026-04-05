#include <Servo.h>
#include <SoftwareSerial.h>

//Arm Servo and Pin Intializations

SoftwareSerial masterSerial(3,2); // RX3 TX2

const int basePin = 7; //placeholder value
const int shoulderPin = 8; //placeholder value
const int elbowPin = 9; //placeholder value

Servo baseServo;
Servo shoulderServo;
Servo elbowServo;


int baseAngle = 90;
int shoulderAngle = 90;
int elbowAngle = 90;


//min and max values
const int baseMin = 0;
const int baseMax = 180;

const int shoulderMin = 30;
const int shoulderMax = 150;

const int elbowMin = 20;
const int elbowMax = 160;

void setup() {
  // put your setup code here, to run once:

  Serial.begin(9600);
  baseServo.attach(basePin);
  shoulderServo.attach(shoulderPin);
  elbowServo.attach(elbowPin);

  baseServo.write(baseAngle);
  shoulderServo.write(shoulderAngle);
  elbowServo.write(elbowAngle);

}

void loop() {
  // put your main code here, to run repeatedly:


  if(masterSerial.available()){
    String cmd = Serial.readStringUntil('\n');

    cmd.trim();

    if(cmd.startsWith("B:")){
      baseAngle = constrain(cmd.substring(2).toInt(),baseMin,baseMax);
      baseServo.write(baseAngle);
    }else if(cmd.startsWith("S:")){
      shoulderAngle = constrain(cmd.substring(2).toInt(),shoulderMin,shoulderMax);
      shoulderServo.write(shoulderAngle);
    }else if(cmd.startsWith("E:")){
      elbowAngle = constrain(cmd.substring(2).toInt(),elbowMin,elbowMax);
      elbowServo.write(elbowAngle);
    }
  }
}
