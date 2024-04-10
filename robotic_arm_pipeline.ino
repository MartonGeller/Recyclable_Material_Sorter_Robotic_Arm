#include <Servo.h>

Servo servo1; // First servo object
Servo servo2; // Second servo object
Servo servo3; // Third servo object
Servo servo5;
Servo servo6;

const int trigPin = 4; // Ultrasonic sensor trigger pin
const int echoPin = 3; // Ultrasonic sensor echo pin
long duration;
float distance;
int currentPos1 = 0;
int currentPos2 = 0;
int currentPos3 = 0;
int currentPos5 = 0;
int t1 = 0;
int t2 = 0;
int t3 = 0;
int t5 = 0; // New servo angle
bool objectSpotted = false;
int target_angle = 0;
float delta = 1.0; // Initial step size for servo movement

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  
  servo1.attach(5); // Attach servo1 to pin 5
  servo2.attach(6); // Attach servo2 to pin 6
  servo3.attach(7);
  servo5.attach(9); // Attach servo3 to pin 7
  servo6.attach(10);
  Serial.begin(9600);
  
  servo1.write(0);
  servo2.write(60);
  servo3.write(150);
  servo5.write(20);
  servo6.write(90);
}

void loop() {
  if (Serial.available() > 0) {
    String message = Serial.readStringUntil('\n');
    message.trim();
    
    if (message.equals("Object has been spotted")) {
      objectSpotted = true;
      // Wait for servo angles
      servo2.write(40);
      delay(1000);
      servo3.write(130);
      delay(10);
      objectSpottedMeasure(); // Start measuring distance
      delay(3000); // Give some time to stop measuring distance
      target_angle = servo1.read();
      delay(2000);
      receiveServoAngles();
        // Wait for servo angles
      
    }
    else if (message.equals("Send")) {
      objectSpotted = false; // Indicate that no object has been spotted// Stop measuring distance
      delay(100); // Give some time to stop measuring distance
      receiveServoAngles(); // Wait for servo angles
      return;
    }
    else {
      target_angle = message.toInt();
      objectSpotted = false;
    }
  }
  
  if (!objectSpotted) {
    uh();

    if (currentPos1 != target_angle) {
      int newPos1 = currentPos1;      

      if (abs(newPos1 - target_angle) > 10) {
        delta = 4.0;
      } else {
        delta = 2.0;
      }

      if (currentPos1 < target_angle) {
        newPos1 += delta;
        delay(25);
      } else if (currentPos1 > target_angle) {
        newPos1 -= delta;
        delay(25);
      }
      
      servo1.write(newPos1);
      currentPos1 = newPos1;
      
      if (abs(newPos1 - target_angle) < 1) {
        Serial.println("Target reached");
      }
    }
  }
}

void receiveServoAngles() {
  Serial.println("Waiting for servo angles");
  delay(1000);
  String t1_str = Serial.readStringUntil('\n');
  t1 = t1_str.toInt();
  Serial.print("Servo1 angle: ");
  Serial.println(t1);

  String t2_str = Serial.readStringUntil('\n');
  t2 = t2_str.toInt();
  Serial.print("Servo2 angle: ");
  Serial.println(t2);

  String t3_str = Serial.readStringUntil('\n');
  t3 = t3_str.toInt();
  Serial.print("Servo3 angle: ");
  Serial.println(t3);

  String t5_str = Serial.readStringUntil('\n');
  t5 = t5_str.toInt();
  Serial.print("Servo5 angle: ");
  Serial.println(t5);
  
  delay(1000);
  pickUp(); // Call the pickUp function with servo angles
  homeState();
}

void uh() {      
  digitalWrite(trigPin, LOW);
  delayMicroseconds(5);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(15);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);
  distance = duration * 0.034/ 2;
  Serial.print("Distance: ");
  Serial.println(distance);
  int currentPos1 = servo1.read();
  Serial.println(currentPos1);
  int currentPos2 = servo2.read();
  Serial.println(currentPos2);
  int currentPos3 = servo3.read();
  Serial.println(currentPos3);
  int currentPos5 = servo5.read();
  Serial.println(currentPos5);
  delay(25);
}

void pickUp(){
  servo6.write(90);
  delay(500);
  servo1.write(t1); 
  Serial.println("Moved servo1");
  delay(1000);
  servo2.write(t2);
  Serial.println("Moved servo2");
  delay(1000);
  servo3.write(t3);
  Serial.println("Moved servo3");
  delay(1000); // Add delay before moving servo4
  servo5.write(t5);
  Serial.println("Moved servo5");
  delay(2000);
  servo6.write(0); // grabs object
  delay(2000);
  dropOff();
  Serial.println("Done moving");
}

void dropOff(){
  delay(1000);
  servo1.write(90);
  delay(1000);
  servo2.write(140);
  delay(1000);
  servo3.write(40);
  delay(1000);
  servo5.write(20);
  delay(1000);
  servo6.write(90); //open gripper
  Serial.println("Object dropped off");
  delay(2000);

}

void homeState(){

  servo1.write(0);
  delay(1000);
  servo2.write(60);
  delay(1000);
  servo3.write(150);
  delay(1000);
  servo5.write(20);
  delay(1000);
  servo6.write(90);
  Serial.println("Moved to homestate");
  }

void objectSpottedMeasure(){
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
  uh();
}