#!/usr/bin/env python3
"""
PM100 Power Meter Reader
Reads power values from PM100 power meter with wavelength input from 400-1300nm.
Displays results every 0.2 seconds.
"""

import time
import sys
from typing import Optional
import threading
import signal

try:
    import pyvisa
except ImportError:
    print("pyvisa library is not installed.")
    print("Please install it using: pip install pyvisa")
    sys.exit(1)

class PM100PowerMeter:
    """PM100 Power Meter control class"""
    
    def __init__(self):
        self.rm: Optional[pyvisa.ResourceManager] = None
        self.instrument: Optional[pyvisa.Resource] = None
        self.wavelength: float = 488.0  # Default wavelength
        self.is_running: bool = False
        self.measurement_thread: Optional[threading.Thread] = None
        
    def connect(self) -> bool:
        """Connect to PM100 device"""
        try:
            self.rm = pyvisa.ResourceManager()
            # Search for USB connected PM100
            resources = self.rm.list_resources()
            pm100_resources = [r for r in resources if 'USB' in r or 'PM100' in r]
            
            if not pm100_resources:
                print("PM100 device not found.")
                print("Please check if the device is connected.")
                return False
            
            # Connect to the first found PM100 device
            self.instrument = self.rm.open_resource(pm100_resources[0])
            self.instrument.timeout = 1000  # Timeout setting
            
            # Get device information
            try:
                idn = self.instrument.query('*IDN?')
                print(f"Connected device: {idn.strip()}")
            except:
                print("Connected to PM100 device.")
            
            return True
            
        except Exception as e:
            print(f"PM100 connection error: {e}")
            return False
    
    def set_wavelength(self, wavelength: float) -> bool:
        """Set laser wavelength"""
        if not (400 <= wavelength <= 1300):
            print("Error: Wavelength must be between 400-1300nm.")
            return False
        
        self.wavelength = wavelength
        
        if self.instrument:
            try:
                # Set wavelength (adjust command based on PM100 model)
                self.instrument.write(f':SENS:CORR:WAV {wavelength}')
                print(f"Wavelength set to {wavelength}nm.")
                return True
            except Exception as e:
                print(f"Wavelength setting error: {e}")
                return False
        else:
            print("Device not connected.")
            return False
    
    def get_power(self) -> Optional[float]:
        """Get current power value in mW"""
        if not self.instrument:
            return None
        
        try:
            # Get power value
            power_str = self.instrument.query(':READ?')
            power_value = float(power_str.strip())
            return power_value
        except Exception as e:
            print(f"Power reading error: {e}")
            return None
    
    def start_continuous_measurement(self):
        """Start continuous measurement"""
        if self.is_running:
            print("Measurement already running.")
            return
        
        self.is_running = True
        self.measurement_thread = threading.Thread(target=self._measurement_loop)
        self.measurement_thread.daemon = True
        self.measurement_thread.start()
        print("Continuous measurement started. Press Ctrl+C to stop.")
    
    def stop_continuous_measurement(self):
        """Stop continuous measurement"""
        self.is_running = False
        if self.measurement_thread:
            self.measurement_thread.join()
        print("\nMeasurement stopped.")
    
    def _measurement_loop(self):
        """Measurement loop (runs in separate thread)"""
        print(f"\nWavelength: {self.wavelength}nm")
        print("Time\t\tPower (mW)")
        print("-" * 30)
        
        start_time = time.time()
        
        while self.is_running:
            try:
                power = self.get_power()
                if power is not None:
                    elapsed_time = time.time() - start_time
                    power_mw = power*1000
                    
                    print(f"{elapsed_time:.1f}s\t\t{power_mw:.2f}")
                else:
                    print("Power reading failed")
                
                time.sleep(0.2)  # 0.2 second interval
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Measurement error: {e}")
                break
    
    def disconnect(self):
        """Disconnect from device"""
        self.stop_continuous_measurement()
        
        if self.instrument:
            self.instrument.close()
            self.instrument = None
        
        if self.rm:
            self.rm.close()
            self.rm = None
        
        print("Disconnected from PM100.")

def signal_handler(sig, frame):
    """Ctrl+C handler"""
    print("\nExiting program...")
    sys.exit(0)

def main():
    """Main function"""
    # Set Ctrl+C signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("PM100 Power Meter Reader")
    print("=" * 40)
    
    # Create PM100 instance
    pm100 = PM100PowerMeter()
    
    try:
        # Connect to device
        if not pm100.connect():
            return
        
        # Input wavelength
        while True:
            try:
                wavelength_input = input("\nEnter laser wavelength (400-1300nm): ")
                wavelength = float(wavelength_input)
                
                if pm100.set_wavelength(wavelength):
                    break
                else:
                    print("Invalid wavelength. Please try again.")
                    
            except ValueError:
                print("Please enter a number.")
            except KeyboardInterrupt:
                print("\nExiting program.")
                return
        
        # Start continuous measurement
        pm100.start_continuous_measurement()
        
        # Wait until user presses Ctrl+C
        try:
            while pm100.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        
    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        # Cleanup
        pm100.disconnect()

if __name__ == "__main__":
    main()