import subprocess
import sys
import os
from pathlib import Path
from typing import Optional


class RTABMapClient:
    """
    A client class to launch RTAB-Map rgbd_playback with specified data directory.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the RTABMapClient with a data directory path.
        
        Args:
            data_dir: Path to the data directory containing processed RGBD data.
        """
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

    def launch(self, use_new_terminal: bool = False, capture_output: bool = False) -> subprocess.Popen:
        """
        Launch the rgbd_playback with the specified data directory.
        
        Args:
            use_new_terminal: If True, launch in a new terminal window. 
                            If False, run in the current process.
            capture_output: If True, capture stdout/stderr to allow monitoring output.
                            Only valid when use_new_terminal=False.
        
        Returns:
            The process object. You can call wait() on it to wait for completion,
            or proceed with other tasks and wait later.
        """
        command = [
            "ros2",
            "launch",
            "rgbd_playback",
            "play_map_mask.launch.py",
            f"data_dir:={self.data_dir}"
        ]
        
        if use_new_terminal:
            if capture_output:
                print("Warning: capture_output is ignored when use_new_terminal=True")
            return self._launch_in_new_terminal(command)
        else:
            return self._launch_in_current_process(command, capture_output)

    def _launch_in_new_terminal(self, command: list) -> subprocess.Popen:
        """
        Launch the command in a new terminal window.
        
        Args:
            command: List of command arguments.
        
        Returns:
            The process object of the terminal.
        """
        # Try different terminal emulators based on the platform
        if sys.platform == "linux" or sys.platform == "linux2":
            # Try common Linux terminal emulators in order of preference
            terminal_commands = [
                ["gnome-terminal", "--", "bash", "-c"],  # GNOME Terminal
                ["konsole", "-e"],                         # KDE Konsole
                ["xterm", "-e"],                           # XTerm
                ["xfce4-terminal", "--execute"],           # Xfce Terminal
                ["tilix", "-e"],                           # Tilix
            ]
            
            for term_cmd in terminal_commands:
                try:
                    full_command = term_cmd + [" ".join(command) + "; bash"]
                    process = subprocess.Popen(full_command)
                    print(f"Launched in new terminal using: {term_cmd[0]}")
                    return process
                except FileNotFoundError:
                    continue
            
            # Fallback: try with bash -i
            try:
                full_command = ["bash", "-i", "-c", " ".join(command)]
                process = subprocess.Popen(full_command)
                print("Launched with bash")
                return process
            except Exception as e:
                raise RuntimeError(f"Failed to launch in new terminal: {e}")
        
        elif sys.platform == "darwin":  # macOS
            try:
                script = f'tell application "Terminal" to do script "{" ".join(command)}"'
                process = subprocess.Popen(["osascript", "-e", script])
                print("Launched in new Terminal (macOS)")
                return process
            except Exception as e:
                raise RuntimeError(f"Failed to launch on macOS: {e}")
        
        elif sys.platform == "win32":  # Windows
            try:
                process = subprocess.Popen(
                    ["cmd", "/c", " ".join(command)],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                print("Launched in new terminal (Windows)")
                return process
            except Exception as e:
                raise RuntimeError(f"Failed to launch on Windows: {e}")
        
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")

    def _launch_in_current_process(self, command: list, capture_output: bool = False) -> subprocess.Popen:
        """
        Launch the command in the current process and return the process object.
        
        Args:
            command: List of command arguments.
            capture_output: Whether to capture stdout/stderr.
        
        Returns:
            The process object.
        """
        print(f"Launching: {' '.join(command)}")
        if capture_output:
            # bufsize=1 means line buffered, text=True means string output
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        else:
            process = subprocess.Popen(command)
        return process

    def wait_for_text(self, process: subprocess.Popen, text: str, timeout: Optional[float] = None) -> int:
        """
        Monitor process output and terminate when specific text is found.
        
        Args:
            process: The process object. Must be launched with capture_output=True.
            text: The text string to look for.
            timeout: Maximum time to wait in seconds.
            
        Returns:
            The return code of the process (or -1 if terminated).
        """
        if not process.stdout:
            raise RuntimeError("Process was not launched with capture_output=True")
            
        print(f"Waiting for text '{text}' in output...")
        import time
        start_time = time.time()
        
        try:
            # Read line by line
            for line in process.stdout:
                print(line, end='') # Echo to console
                
                if text in line:
                    print(f"\nDetected '{text}'. Terminating process...")
                    self.terminate(process)
                    return -1
                
                if timeout and (time.time() - start_time > timeout):
                    print(f"\nTimeout ({timeout}s) reached while waiting for text.")
                    self.terminate(process)
                    return -1
                    
        except KeyboardInterrupt:
            print("\nProcess stopped by user.")
            self.terminate(process)
            return -1
            
        # If process ends without finding text
        return process.returncode

    def wait(self, process: subprocess.Popen, timeout: Optional[float] = None) -> int:
        """
        Wait for the process to complete with optional timeout.
        
        Args:
            process: The process object returned from launch().
            timeout: Maximum time to wait in seconds. If set and exceeded, 
                    the process will be terminated. None means wait indefinitely.
        
        Returns:
            The return code of the process.
        """
        try:
            returncode = process.wait(timeout=timeout)
            print(f"Process completed with return code: {returncode}")
            return returncode
        except subprocess.TimeoutExpired:
            print(f"\nTimeout ({timeout}s) reached. Terminating process...")
            self.terminate(process)
            return -1
        except KeyboardInterrupt:
            print("\nProcess stopped by user.")
            self.terminate(process)
            return -1

    def terminate(self, process: subprocess.Popen, timeout: float = 5) -> None:
        """
        Gracefully terminate the process and all its child processes.
        
        Args:
            process: The process object to terminate.
            timeout: Time to wait for graceful termination before force killing.
        """
        if process.poll() is not None:
            # Process already terminated
            return
        
        print(f"Sending SIGTERM to process {process.pid}...")
        process.terminate()
        
        try:
            process.wait(timeout=timeout)
            print("Process terminated gracefully.")
        except subprocess.TimeoutExpired:
            print(f"Process did not terminate within {timeout}s, killing it...")
            process.kill()
            process.wait()
            print("Process killed.")

    def kill(self, process: subprocess.Popen) -> None:
        """
        Force kill the process immediately.
        
        Args:
            process: The process object to kill.
        """
        if process.poll() is None:
            print(f"Force killing process {process.pid}...")
            process.kill()
            process.wait()
            print("Process killed.")


# Example usage
if __name__ == "__main__":
    # Example 1: Launch and wait for specific output text
    data_dir = "/home/yutian/Hand2Gripper_phantom/data/processed/epic/0"
    
    try:
        client = RTABMapClient(data_dir)
        
        # Launch with capture_output=True to enable text monitoring
        process = client.launch(use_new_terminal=False, capture_output=True)
        print(f"Process started with PID: {process.pid}")
        
        # Wait until "finished" appears in the output
        # Based on your log: "[INFO] [player_mapper_masker-2]: process has finished cleanly"
        # or "播放结束"
        target_text = ">>> VSLAM PLAYBACK COMPLETE <<<" 
        
        client.wait_for_text(process, target_text, timeout=300)
        
        print("Process finished! Proceeding with next tasks...")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Error: {e}")
