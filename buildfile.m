function plan = buildfile
import matlab.buildtool.tasks.*

plan = buildplan(localfunctions);

plan("clean") = CleanTask;
plan("check") = CodeIssuesTask;
plan("test") = TestTask;

plan.DefaultTasks = ["check" "Build" "test"];
end


function BuildTask(context)
% Run a specific file using its explicit path
run("vdgp_setup.m");
run("vdgp_build_mex.m");
run("test_vdgp.m")
end
