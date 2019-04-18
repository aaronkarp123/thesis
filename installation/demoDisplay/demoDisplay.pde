import oscP5.*;
import netP5.*;

OscP5 oscP5;
NetAddress remoteLocation;

int rectX, rectY;      // Position of square button
int rectSize = 90;     // Diameter of rect
color rectColorOn, rectColorOff, baseColor;
color rectHighlightOn, rectHighlightOff;
color currentColor;
boolean rectOver = false;
boolean on = false;

void setup() {
  size(800,800);
  frameRate(25);
  /* start oscP5, listening for incoming messages at port 12000 */
  oscP5 = new OscP5(this,500);
  remoteLocation = new NetAddress("127.0.0.1",12000);
  
  
  rectColorOff = color(255,0,0);
  rectColorOn = color(0,255,0);
  rectHighlightOn = color(0,150,0);
  rectHighlightOff = color(150,0,0);
  baseColor = color(102);
  currentColor = baseColor;
  rectX = width/2-rectSize/2;
  rectY = height/2-rectSize/2-50;
  ellipseMode(CENTER);
}


void draw() {
  update(mouseX, mouseY);
  background(currentColor);
  if (rectOver) {
    if (on) {
      fill(rectHighlightOn);
    }
    else {
      fill(rectHighlightOff);
    }
  } else {
    if (on) {
      fill(rectColorOn);
    }
    else {
      fill(rectColorOff);
    }
  }
  stroke(180);
  rect(rectX, rectY, rectSize, rectSize);
  
  textSize(32);
  String dispText = "the Acoustic Counterfeit Machine is Off";
  fill(0,0,0);
  if (on){
    dispText = "the Acoustic Counterfeit Machine is On";
    fill(0,30,60);
  }
  textAlign(CENTER,BOTTOM);
  text(dispText, width/2, height/2+50); 
}

void update(int x, int y) {
  if ( overRect(rectX, rectY, rectSize, rectSize) ) {
    rectOver = true;
  } else {
    rectOver = false;
  }
}


boolean overRect(int x, int y, int width, int height)  {
  if (mouseX >= x && mouseX <= x+width && 
      mouseY >= y && mouseY <= y+height) {
    return true;
  } else {
    return false;
  }
}

void mousePressed() {
  if (rectOver) {
    on = !on;
    /* in the following different ways of creating osc messages are shown by example */
    OscMessage myMessage = new OscMessage("/onoff");
    
    myMessage.add(int(on)); /* add an int to the osc message */
  
    /* send the message */
    oscP5.send(myMessage, remoteLocation); 
    print("SENT");
  }
}


/* incoming osc message are forwarded to the oscEvent method. */
void oscEvent(OscMessage theOscMessage) {
  /* print the address pattern and the typetag of the received OscMessage */
  print("### received an osc message.");
  print(" addrpattern: "+theOscMessage.addrPattern());
  println(" typetag: "+theOscMessage.typetag());
}
