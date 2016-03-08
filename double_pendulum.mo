within ;
model DoublePendulum "MSL Double pendulum"

  extends Modelica.Icons.Example;
  inner Modelica.Mechanics.MultiBody.World world annotation (Placement(
        transformation(extent={{-88,0},{-68,20}}, rotation=0)));
  Modelica.Mechanics.MultiBody.Joints.Revolute revolute1(useAxisFlange=true,
      w(fixed=true),
    phi(
      fixed=true,
      displayUnit="deg",
      start=-1.5707963267949))                                   annotation (Placement(transformation(extent={{-48,0},
            {-28,20}}, rotation=0)));
  Modelica.Mechanics.Rotational.Components.Damper damper(d=0.1)
    annotation (Placement(transformation(extent={{-48,40},{-28,60}}, rotation=0)));
  Modelica.Mechanics.MultiBody.Parts.BodyBox boxBody1(r={0.5,0,0}, width=0.06)
    annotation (Placement(transformation(extent={{-10,0},{10,20}}, rotation=0)));
  Modelica.Mechanics.MultiBody.Joints.Revolute revolute2(                 w(
        fixed=true),
    useAxisFlange=false,
    phi(fixed=true, start=0))                            annotation (Placement(transformation(extent={{32,0},{
            52,20}}, rotation=0)));
  Modelica.Mechanics.MultiBody.Parts.BodyBox boxBody2(r={0.5,0,0}, width=0.06)
    annotation (Placement(transformation(extent={{74,0},{94,20}}, rotation=0)));
  Modelica.Mechanics.MultiBody.Forces.WorldTorque torque1
    annotation (Placement(transformation(extent={{-46,-44},{-26,-24}})));
  Modelica.Blocks.Routing.Multiplex3 multiplex3
    annotation (Placement(transformation(extent={{-70,-40},{-58,-28}})));
  Modelica.Blocks.Interfaces.RealInput u
    annotation (Placement(transformation(extent={{-138,-78},{-108,-48}})));
  Modelica.Blocks.Sources.Constant const(k=0)
    annotation (Placement(transformation(extent={{-124,-16},{-108,0}})));
  Modelica.Blocks.Sources.Constant const1(k=0)
    annotation (Placement(transformation(extent={{-124,-42},{-108,-26}})));
equation

  connect(revolute1.support, damper.flange_a) annotation (Line(points={{-44,20},
          {-44,28},{-58,28},{-58,50},{-48,50}}, color={0,0,0}));
  connect(revolute1.frame_b, boxBody1.frame_a)
    annotation (Line(
      points={{-28,10},{-10,10}},
      color={95,95,95},
      thickness=0.5));
  connect(revolute2.frame_b, boxBody2.frame_a)
    annotation (Line(
      points={{52,10},{74,10}},
      color={95,95,95},
      thickness=0.5));
  connect(boxBody1.frame_b, revolute2.frame_a)
    annotation (Line(
      points={{10,10},{32,10}},
      color={95,95,95},
      thickness=0.5));
  connect(world.frame_b, revolute1.frame_a)
    annotation (Line(
      points={{-68,10},{-48,10}},
      color={95,95,95},
      thickness=0.5));
  connect(damper.flange_b, revolute1.axis) annotation (Line(
      points={{-28,50},{-20,50},{-20,20},{-38,20}},
      color={0,0,0},
      smooth=Smooth.None));
  connect(torque1.frame_b, boxBody1.frame_a) annotation (Line(
      points={{-26,-34},{-20,-34},{-20,10},{-10,10}},
      color={95,95,95},
      thickness=0.5,
      smooth=Smooth.None));
  connect(multiplex3.y, torque1.torque) annotation (Line(
      points={{-57.4,-34},{-48,-34}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(multiplex3.u3[1], u) annotation (Line(
      points={{-71.2,-38.2},{-94,-38.2},{-94,-63},{-123,-63}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(const1.y, multiplex3.u2[1]) annotation (Line(
      points={{-107.2,-34},{-71.2,-34}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(const.y, multiplex3.u1[1]) annotation (Line(
      points={{-107.2,-8},{-94,-8},{-94,-29.8},{-71.2,-29.8}},
      color={0,0,127},
      smooth=Smooth.None));
  annotation (
    experiment(StopTime=3),
    Documentation(info="<html>
<p>
This example demonstrates that by using joint and body
elements animation is automatically available. Also the revolute
joints are animated. Note, that animation of every component
can be switched of by setting the first parameter <b>animation</b>
to <b>false</b> or by setting <b>enableAnimation</b> in the <b>world</b>
object to <b>false</b> to switch off animation of all components.
</p>

<table border=0 cellspacing=0 cellpadding=0><tr><td valign=\"top\">
<IMG src=\"modelica://Modelica/Resources/Images/Mechanics/MultiBody/Examples/Elementary/DoublePendulum.png\"
ALT=\"model Examples.Elementary.DoublePendulum\">
</td></tr></table>

</HTML>"),
    uses(Modelica(version="3.2.1")),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-140,-100},{
            100,100}}), graphics),
    Icon(coordinateSystem(extent={{-140,-100},{100,100}})));
end DoublePendulum;
