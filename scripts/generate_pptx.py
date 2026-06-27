import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def create_presentation():
    prs = Presentation()
    
    # Set slide dimensions to widescreen 16:9
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    
    # Colors
    c_navy = RGBColor(12, 35, 64)
    c_blue = RGBColor(30, 144, 255)
    c_dark = RGBColor(50, 50, 50)
    c_gray = RGBColor(128, 128, 128)
    
    # Helper to style text frames
    def set_font(run, size=18, bold=False, italic=False, color=c_dark, font_name="Calibri"):
        run.font.name = font_name
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color

    # Helper to add standard slide with Title
    def add_slide(title_text):
        blank_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(blank_layout)
        
        # Add Title Box
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.33), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_bottom = tf.margin_right = 0
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Calibri"
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = c_navy
        
        return slide

    # Slide 1: Title Slide
    slide1 = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Background Accent shape
    accent = slide1.shapes.add_shape(
        1, Inches(0), Inches(0), Inches(4.5), Inches(7.5)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = c_navy
    accent.line.color.rgb = c_navy
    
    title_box = slide1.shapes.add_textbox(Inches(5.0), Inches(2.0), Inches(7.8), Inches(4.0))
    tf = title_box.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "Reinforcement Learning for Underwater Acoustic Vessel Detection and Tracking"
    set_font(p.runs[0], size=36, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "\nCourse: Introduction to Reinforcement Learning"
    set_font(p2.runs[0], size=20, color=c_blue)
    
    p3 = tf.add_paragraph()
    p3.text = "Author: Roy Michael\nDate: June 2026"
    set_font(p3.runs[0], size=18, color=c_gray)
    
    notes1 = slide1.notes_slide.notes_text_frame
    notes1.text = (
        "Welcome everyone to this presentation on Reinforcement Learning for Underwater Acoustic "
        "Vessel Detection and Tracking.\n\n"
        "In this talk, we will discuss how to formulate passive sonar tracking as a Markov Decision "
        "Process, and evaluate Double Q-Learning, Linear Function Approximation (Tile Coding), "
        "and Actor-Critic policies on real-world ocean datasets collected in Haifa and Croatia."
    )

    # Slide 2: Project Overview & Motivation
    slide2 = add_slide("Project Overview & Motivation")
    tb = slide2.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.33), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "• The Challenge of Passive Sonar Tracking:"
    set_font(p1.runs[0], size=20, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "  - Marine environments are filled with non-stationary background noise (wave clutter, biological clicks)."
    set_font(p2.runs[0], size=18, color=c_dark)
    
    p3 = tf.add_paragraph()
    p3.text = "  - Vessel acceleration and maneuvers induce large frequency shifts (Doppler drifts) and temporary fading."
    set_font(p3.runs[0], size=18, color=c_dark)
    
    p4 = tf.add_paragraph()
    p4.text = "• Limitations of Heuristic Trackers:"
    set_font(p4.runs[0], size=20, bold=True, color=c_navy)
    p4.space_before = Pt(20)
    
    p5 = tf.add_paragraph()
    p5.text = "  - Traditional rule-based trackers use rigid gates. Narrow gates lose tracks during acceleration; wide gates allow false associations with noise."
    set_font(p5.runs[0], size=18, color=c_dark)
    
    p6 = tf.add_paragraph()
    p6.text = "• RL Formulation Goal:"
    set_font(p6.runs[0], size=20, bold=True, color=c_navy)
    p6.space_before = Pt(20)
    
    p7 = tf.add_paragraph()
    p7.text = "  - Automate tracking decisions as an MDP, balancing track continuity with noise rejection to maximize long-term rewards."
    set_font(p7.runs[0], size=18, color=c_dark)

    notes2 = slide2.notes_slide.notes_text_frame
    notes2.text = (
        "Here we introduce the motivation for our work.\n\n"
        "Passive sonar processing is inherently difficult because the underwater channel is highly non-stationary. "
        "We have biological noises, snapping shrimp, wave action, and wind. Traditional systems use heuristically tuned "
        "gating thresholds. If the vessel speeds up, its engine tone slides in frequency due to the Doppler shift. A narrow "
        "threshold will immediately drop the track, while a wide threshold will associate with nearby clutter.\n\n"
        "We want to reformulate this as a sequential decision-making process where the agent learns the optimal policy "
        "under uncertainty."
    )

    # Slide 3: Experimental Setup
    slide3 = add_slide("Field Experiments \& Mooring Setup")
    
    # Left text column
    tb = slide3.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(6.0), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "Croatia Ocean Sonics Dataset (Silba Island):"
    set_font(p1.runs[0], size=18, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "• Radiated noise from Suex VRX Diver Propulsion Vehicles (DPVs) and surface support boats.\n• Real-world interference including passing ferries."
    set_font(p2.runs[0], size=16, color=c_dark)
    
    p3 = tf.add_paragraph()
    p3.text = "Haifa Dataset:"
    set_font(p3.runs[0], size=18, bold=True, color=c_navy)
    p3.space_before = Pt(15)
    
    p4 = tf.add_paragraph()
    p4.text = "• Seacraft GO! DVP executing circular and linear maneuvers."
    set_font(p4.runs[0], size=16, color=c_dark)
    
    p5 = tf.add_paragraph()
    p5.text = "Deployment Equipment:"
    set_font(p5.runs[0], size=18, bold=True, color=c_navy)
    p5.space_before = Pt(15)
    
    p6 = tf.add_paragraph()
    p6.text = "• icListen HF ALTA hydrophone unit (128 kHz sampling rate).\n• Moored on seabed frames at 10-30m depths to eliminate surface waves and buoy movements."
    set_font(p6.runs[0], size=16, color=c_dark)
    
    # Right Image column
    if os.path.exists("report/images/silba-1k.png"):
        slide3.shapes.add_picture("report/images/silba-1k.png", Inches(7.0), Inches(1.8), width=Inches(5.5))
        
    notes3 = slide3.notes_slide.notes_text_frame
    notes3.text = (
        "To gather realistic acoustic signatures, field trials were performed in Haifa, Israel and Silba Island, Croatia.\n\n"
        "The targets of interest are Diver Propulsion Vehicles, also known as underwater scooters, which emit quiet, "
        "low-frequency narrowband motor tonals from their pulse-width modulated electric drives. Support boats and ferries "
        "are also recorded, representing typical harbor noise clutter.\n\n"
        "Crucially, the hydrophone is placed on a seabed-anchored frame. This isolates the hydrophone from the movement of the "
        "surface waves and cable strumming, which would otherwise introduce massive low-frequency noise."
    )

    # Slide 3b: Intro to Underwater Acoustics
    slide3b = add_slide("Introduction to Underwater Acoustics")
    tb = slide3b.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.33), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "Acoustic Propagation in Saline Water:"
    set_font(p1.runs[0], size=20, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "• Electromagnetic waves suffer from extreme attenuation in saline water.\n" \
             "• Sound waves propagate efficiently (c ~ 1500 m/s), serving as the primary modality for underwater sensing."
    set_font(p2.runs[0], size=16, color=c_dark)
    
    p3 = tf.add_paragraph()
    p3.text = "Passive Sonar & LOFAR Spectrograms:"
    set_font(p3.runs[0], size=20, bold=True, color=c_navy)
    p3.space_before = Pt(15)
    
    p4 = tf.add_paragraph()
    p4.text = "• Hydrophones capture continuous pressure signals, converted to the time-frequency domain via STFT.\n" \
             "• Broadband Background Noise: Flatly distributed spectral clutter (wind, waves, distant traffic).\n" \
             "• Narrowband Tonals: Sharp, persistent lines matching engine/machinery rotation frequencies."
    set_font(p4.runs[0], size=16, color=c_dark)
    
    notes3b = slide3b.notes_slide.notes_text_frame
    notes3b.text = (
        "Let's step back and look at the physics of underwater acoustics.\n\n"
        "Radio and light waves do not travel far in water due to high conductivity and absorption. "
        "Therefore, we use acoustic waves. Passive sonar listens to the acoustic energy radiated by a target.\n\n"
        "The signal is displayed on a LOFAR spectrogram. We separate the incoming signals into flat, broadband noise "
        "and discrete narrowband tonals, which represent machinery components like diesel engines or generators."
    )

    # Slide 3c: Acoustic Channel Challenges
    slide3c = add_slide("Physical Challenges in Sonar Tracking")
    tb = slide3c.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.33), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "1. Doppler Frequency Shift:"
    set_font(p1.runs[0], size=18, bold=True, color=c_navy)
    p2 = tf.add_paragraph()
    p2.text = "   - Target velocity changes induce shifts in observed tonals: f_obs = f_0 * (c +/- v_r)/(c -/+ v_s).\n" \
             "   - Accelerating vessels create 'sliding' frequency tracks."
    set_font(p2.runs[0], size=16, color=c_dark)
    
    p3 = tf.add_paragraph()
    p3.text = "2. Multipath Propagation & Echoes:"
    set_font(p3.runs[0], size=18, bold=True, color=c_navy)
    p3.space_before = Pt(15)
    p4 = tf.add_paragraph()
    p4.text = "   - Reflections off the seabed and surface create duplicate tracking paths and harmonic overtones."
    set_font(p4.runs[0], size=16, color=c_dark)
    
    p5 = tf.add_paragraph()
    p5.text = "3. Signal Fading & Attenuation:"
    set_font(p5.runs[0], size=18, bold=True, color=c_navy)
    p5.space_before = Pt(15)
    p6 = tf.add_paragraph()
    p6.text = "   - Destructive interference creates Doppler nulls, requiring trackers to maintain state through periods of zero-amplitude feedback."
    set_font(p6.runs[0], size=16, color=c_dark)
    
    notes3c = slide3c.notes_slide.notes_text_frame
    notes3c.text = (
        "Tracking is highly complex due to physical acoustic phenomena.\n\n"
        "First, Doppler shift causes frequency changes when vessels speed up or turn. "
        "Second, multipath reflections cause phantom echoes and harmonics. "
        "Third, acoustic fading can make a tonal completely disappear for a few seconds. "
        "The tracking agent must handle all of these dynamics under uncertainty."
    )

    # Slide 3d: LOFAR Spectrogram Comparison
    slide3d = add_slide("LOFAR Spectrogram Comparison")
    # Left image (Raw)
    if os.path.exists("report/images/LOFAR_Joint_Signal.png"):
        slide3d.shapes.add_picture("report/images/LOFAR_Joint_Signal.png", Inches(0.5), Inches(1.5), width=Inches(6.0))
    # Right image (Annotated)
    if os.path.exists("report/images/lofar_annotated.png"):
        slide3d.shapes.add_picture("report/images/lofar_annotated.png", Inches(6.8), Inches(1.5), width=Inches(6.0))
        
    notes3d = slide3d.notes_slide.notes_text_frame
    notes3d.text = (
        "Here we show the visual spectrograms. On the left is the raw LOFAR spectrogram, showing the continuous shipping noise "
        "and overlapping curves. On the right, we have annotated the base frequencies in green, and their integer multiple "
        "harmonics stacked above in orange and red.\n\n"
        "Our RL agent processes these signals frame-by-frame, deciding whether to track these lines or reject the noise."
    )

    # Slide 4: MDP Formulation & State Space
    slide4 = add_slide("Problem Formulation: MDP State Space")
    
    tb = slide4.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.33), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "Markov Decision Process (MDP) Tuple: <S, A, P, R, gamma>"
    set_font(p1.runs[0], size=22, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "Continuous State Space (S):"
    set_font(p2.runs[0], size=20, bold=True, color=c_navy)
    p2.space_before = Pt(15)
    
    p3 = tf.add_paragraph()
    p3.text = "  s_continuous = (d_Hz, A, T)"
    set_font(p3.runs[0], size=18, bold=True, color=c_blue)
    
    p4 = tf.add_paragraph()
    p4.text = "  • d_Hz : Spectral distance from the candidate peak to the closest active vessel track.\n" \
             "  • A : Relative amplitude (activation weight) of the spectral peak.\n" \
             "  • T : Tonality score (Wiener Entropy-based). Distinguishes machinery tonals from noise."
    set_font(p4.runs[0], size=16, color=c_dark)
    
    p5 = tf.add_paragraph()
    p5.text = "Discretized State Bins (S_discrete) for Tabular Policies:"
    set_font(p5.runs[0], size=20, bold=True, color=c_navy)
    p5.space_before = Pt(15)
    
    p6 = tf.add_paragraph()
    p6.text = "  • Distance Bins: Very close (<= 15 Hz) | Med close (<= 45 Hz) | Far (<= 90 Hz) | Out of range.\n" \
             "  • Amplitude Bins: Low (< 0.005) | Medium (0.005 to 0.02) | High (>= 0.02).\n" \
             "  • Tonality Bins: Noise (< 0.45) | Moderate (0.45 to 0.65) | High (>= 0.65)."
    set_font(p6.runs[0], size=16, color=c_dark)

    notes4 = slide4.notes_slide.notes_text_frame
    notes4.text = (
        "We formalize the tracking problem as an MDP.\n\n"
        "For each incoming acoustic peak detected in a frame, the agent observes a state consisting of three elements: "
        "distance, amplitude, and tonality. The distance relates the new detection to our existing trackers. The amplitude "
        "and tonality measures the signal quality.\n\n"
        "For tabular policies, we discretize these continuous dimensions into a 4x3x3 grid (36 discrete states). "
        "This extremely compact state space is a key design choice that allows Q-Learning to converge very quickly, "
        "as we will see in the results."
    )

    # Slide 5: Exclusion of Vessel ID
    slide5 = add_slide("Architectural Choice: Vessel ID Exclusion")
    
    tb = slide5.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.33), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "Why the Vessel ID is NOT included in the RL State Space:"
    set_font(p1.runs[0], size=22, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "1. Identity Invariance & Generalization:"
    set_font(p2.runs[0], size=18, bold=True, color=c_navy)
    p2.space_before = Pt(15)
    p3 = tf.add_paragraph()
    p3.text = "   - Acoustic tracking rules (Doppler drift, engine tonality) are identical for all vessels. A tracking policy trained on Vessel #1 generalizes perfectly to Vessel #2. Including IDs would prevent generalization to unseen vessels."
    set_font(p3.runs[0], size=16, color=c_dark)
    
    p4 = tf.add_paragraph()
    p4.text = "2. State Space Compactness:"
    set_font(p4.runs[0], size=18, bold=True, color=c_navy)
    p4.space_before = Pt(15)
    p5 = tf.add_paragraph()
    p5.text = "   - Including specific ID tokens would expand the state space infinitely as the number of targets increases, causing severe state sparsity and rendering tabular Q-learning intractable."
    set_font(p5.runs[0], size=16, color=c_dark)
    
    p6 = tf.add_paragraph()
    p6.text = "3. Split Architecture Separation of Concerns:"
    set_font(p6.runs[0], size=18, bold=True, color=c_navy)
    p6.space_before = Pt(15)
    p7 = tf.add_paragraph()
    p7.text = "   - The RL agent acts as a local router, making local decisions based on relative state variables. The DSP Orchestrator acts as the global bookkeeper, managing Vessel IDs externally."
    set_font(p7.runs[0], size=16, color=c_dark)

    notes5 = slide5.notes_slide.notes_text_frame
    notes5.text = (
        "Students often ask why the Vessel ID is not part of the state space. This is a critical design choice in our "
        "hybrid architecture.\n\n"
        "If you include the Vessel ID (e.g. Vessel 1, Vessel 2) in the state, the agent would have to learn how to track "
        "Vessel 1 and Vessel 2 independently. The state space would expand infinitely with each new boat. But the physics of "
        "tracking (distance and frequency continuity) are identical for all boats. Therefore, the state is formulated in "
        "relative coordinates (distance to the closest tracker), and the actual ID management is handled by the wrapper "
        "orchestration layer."
    )

    # Slide 6: Action Space & Decoupled Reward Function
    slide6 = add_slide("Action Space \& Decoupled Reward Rules")
    
    tb = slide6.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.33), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "Action Options (a) and Environment Feedback (Reward r):"
    set_font(p1.runs[0], size=22, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "• REJECT (a = 0): Discard the peak observation."
    set_font(p2.runs[0], size=18, bold=True, color=c_navy)
    p2.space_before = Pt(10)
    p3 = tf.add_paragraph()
    p3.text = "  - Correct Reject (r = +2.0): Ignoring background noise/clutter.\n  - False Negative (r = -10.0): Ignoring a highly tonal, close target peak."
    set_font(p3.runs[0], size=16, color=c_dark)
    
    p4 = tf.add_paragraph()
    p4.text = "• ASSOCIATE (a = 1): Route peak to closest active tracker."
    set_font(p4.runs[0], size=18, bold=True, color=c_navy)
    p4.space_before = Pt(10)
    p5 = tf.add_paragraph()
    p5.text = "  - Good Association (r = +10.0): Centroid distance <= 30 Hz.\n  - Speed Change (r = +5.0): Centroid distance 30 to 65 Hz (splits track into new speed stage).\n  - Bad Mismatch (r = -15.0): distance > 65 Hz.  |  - Invalid Association (r = -20.0): No trackers exist."
    set_font(p5.runs[0], size=16, color=c_dark)
    
    p6 = tf.add_paragraph()
    p6.text = "• SPAWN (a = 2): Spawn a new Vessel Tracker."
    set_font(p6.runs[0], size=18, bold=True, color=c_navy)
    p6.space_before = Pt(10)
    p7 = tf.add_paragraph()
    p7.text = "  - Correct Spawn: +10.0 (High tonality peak) or +5.0 (Medium tonality peak).\n  - Duplicate Spawn (r = -10.0): Creating a new tracker when an active one is already nearby (<= 35 Hz)."
    set_font(p7.runs[0], size=16, color=c_dark)

    notes6 = slide6.notes_slide.notes_text_frame
    notes6.text = (
        "Here we show the core reward matrix of the system.\n\n"
        "At each frame, the agent receives a peak and chooses one of three actions: Reject, Associate, or Spawn.\n\n"
        "The environment resolves this action and assigns a reward. For example, if we Associate a peak that is very close "
        "to an active track (<= 30 Hz), we get a Good Association reward of +10.0. If it is further away (30-65 Hz), we assume "
        "the vessel changed speed and give a +5.0 reward. Spawning on noise or spawning a duplicate tracker near an existing "
        "one is penalized. Attempting to associate when no trackers have been spawned yet is the most severe penalty, at -20.0."
    )

    # Slide 7: Rationale for Reward Magnitudes
    slide7 = add_slide("Reward Design: Preventing Degenerate Policies")
    
    tb = slide7.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.33), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "Reward engineering was utilized to avoid critical exploit policies:"
    set_font(p1.runs[0], size=22, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "1. Preventing the 'Lazy' Agent Policy:"
    set_font(p2.runs[0], size=18, bold=True, color=c_navy)
    p2.space_before = Pt(15)
    p3 = tf.add_paragraph()
    p3.text = "   - If the reward for a Correct Reject was too high, the agent would learn a degenerate policy of rejecting every peak to safely accumulate points, avoiding any risk of association penalties.\n" \
             "   - Solution: Correct Reject reward is a small baseline (+2.0) while Good Association is five times higher (+10.0)."
    set_font(p3.runs[0], size=16, color=c_dark)
    
    p4 = tf.add_paragraph()
    p4.text = "2. Preventing 'Spawn Farming' Exploits:"
    set_font(p4.runs[0], size=18, bold=True, color=c_navy)
    p4.space_before = Pt(15)
    p5 = tf.add_paragraph()
    p5.text = "   - If spawning a tracker yielded equal or higher reward than maintaining it, the agent would constantly drop active tracks and spawn new ones to harvest points.\n" \
             "   - Solution: Moderate Spawn reward is capped at +5.0, keeping it lower than Good Association (+10.0)."
    set_font(p5.runs[0], size=16, color=c_dark)

    notes7 = slide7.notes_slide.notes_text_frame
    notes7.text = (
        "Designing the reward system in reinforcement learning is often a process of trial and error because agents are "
        "extremely good at finding loopholes or degenerate policies.\n\n"
        "For example, when we first set the rewards, the agent learned a 'lazy policy.' Because associating a peak carries the "
        "risk of a -15.0 mismatch penalty, and rejecting noise gives a positive reward, the agent realized it could maximize its "
        "score by simply rejecting every peak it saw, never starting any tracks. We solved this by scaling the association "
        "reward to be five times larger than the reject reward.\n\n"
        "Similarly, we had to moderate the spawn reward to prevent 'spawn farming' where the agent would spawn a tracker and "
        "immediately drop it to spawn another."
    )

    # Slide 8: Self-Supervised Label-Free Training
    slide8 = add_slide("Self-Supervised Label-Free Training Dynamics")
    
    tb = slide8.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.33), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "How the agent trains autonomously without human labels:"
    set_font(p1.runs[0], size=22, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "• Constraint-Based Validation (The Environment Critic):"
    set_font(p2.runs[0], size=18, bold=True, color=c_navy)
    p2.space_before = Pt(15)
    p3 = tf.add_paragraph()
    p3.text = "  - The environment enforces physical consistency. For instance, spawning multiple trackers close together (<= 35 Hz) is redundant (Duplicate Spawn, -10.0). Associating a peak that is too far away (> 65 Hz) violates spectral continuity (-15.0). The physics of acoustic propagation defines correctness."
    set_font(p3.runs[0], size=16, color=c_dark)
    
    p4 = tf.add_paragraph()
    p4.text = "• Temporal Consistency (TD Learning):"
    set_font(p4.runs[0], size=18, bold=True, color=c_navy)
    p4.space_before = Pt(15)
    p5 = tf.add_paragraph()
    p5.text = "  - Spawning a tracker on a transient noise spike might not cause an immediate penalty. However, in the next steps, the noise fades, the track gets no associations, ages, and eventually times out. Through Temporal Difference updates, this future failure is propagated back to the initial spawn decision, training the agent to ignore transients."
    set_font(p5.runs[0], size=16, color=c_dark)

    notes8 = slide8.notes_slide.notes_text_frame
    notes8.text = (
        "A major highlight of this project is that it requires zero labeled training data. There is no human annotator "
        "marking where the boats are.\n\n"
        "Correctness is enforced through physical constraints in the environment critic (e.g. redundancy and continuity bounds) "
        "and temporal consistency.\n\n"
        "If the agent decides to spawn a tracker on a transient bubble or dolphin click, it might not look bad in that single frame. "
        "However, because the bubble is a transient event, it will disappear. The tracker will then sit empty, accumulate age, "
        "and time out, hitting the agent with future penalties. Temporal Difference learning propagates this future failure "
        "back to the initial spawn action, teaching the agent to avoid spawning on non-tonal transients."
    )

    # Slide 9: RL-DSP System Architecture
    slide9 = add_slide("RL-DSP System Architecture")
    
    # Left text column
    tb = slide9.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(6.0), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "Separation of Concerns:"
    set_font(p1.runs[0], size=18, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "• core.environment Package:"
    set_font(p2.runs[0], size=16, bold=True, color=c_navy)
    p3 = tf.add_paragraph()
    p3.text = "  - AcousticDataStreamer streams spectral frames.\n  - TrackingMDPEnv wraps streamer, computes state, resolves rewards."
    set_font(p3.runs[0], size=14, color=c_dark)
    
    p4 = tf.add_paragraph()
    p4.text = "• core.agent Package:"
    set_font(p4.runs[0], size=16, bold=True, color=c_navy)
    p4.space_before = Pt(10)
    p5 = tf.add_paragraph()
    p5.text = "  - DSPOrchestrator manages NMF updates, extracts peaks, routes peaks to dynamic trackers.\n  - VesselTrackProcessor tracks independent vessels."
    set_font(p5.runs[0], size=14, color=c_dark)
    
    # Right Image column
    if os.path.exists("report/images/high_level_architecture.png"):
        slide9.shapes.add_picture("report/images/high_level_architecture.png", Inches(6.5), Inches(1.8), width=Inches(6.3))

    notes9 = slide9.notes_slide.notes_text_frame
    notes9.text = (
        "Here we look at the system architecture.\n\n"
        "We divide the code strictly into two packages: core.environment and core.agent.\n\n"
        "In the environment package, we run the low-level signal processing like the STFT frame buffer and the standard Gym-like "
        "MDP wrapper. The agent package houses the orchestrator, which runs the NMF model on the STFT buffer, extracts candidate "
        "peak frequencies, and manages dynamically spawned vessel track processors.\n\n"
        "This boundary isolates the deterministic physical processing (NMF and clustering) from the stochastically updated "
        "reinforcement learning policy."
    )

    # Slide 10: Evaluated Algorithms
    slide10 = add_slide("Evaluated Reinforcement Learning Algorithms")
    
    tb = slide10.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.33), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "We evaluated three distinct learning paradigms:"
    set_font(p1.runs[0], size=22, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "1. Tabular Double Q-Learning:"
    set_font(p2.runs[0], size=18, bold=True, color=c_navy)
    p2.space_before = Pt(15)
    p3 = tf.add_paragraph()
    p3.text = "   - Maintains two independent tables (Q_A, Q_B) to decouple action selection from evaluation.\n   - Removes maximization bias, preventing the agent from chasing random noise spikes."
    set_font(p3.runs[0], size=16, color=c_dark)
    
    p4 = tf.add_paragraph()
    p4.text = "2. Linear Function Approximation (Tile Coding):"
    set_font(p4.runs[0], size=18, bold=True, color=c_navy)
    p4.space_before = Pt(15)
    p5 = tf.add_paragraph()
    p5.text = "   - Maps continuous coordinates (d_Hz, A, T) to a 2,048-dimensional sparse binary feature space using 4 overlapping tilings.\n   - Avoids boundary discretization artifacts, generalizing smoothly across Doppler drifts."
    set_font(p5.runs[0], size=16, color=c_dark)
    
    p6 = tf.add_paragraph()
    p6.text = "3. Actor-Critic (Stochastic Policy):"
    set_font(p6.runs[0], size=18, bold=True, color=c_navy)
    p6.space_before = Pt(15)
    p7 = tf.add_paragraph()
    p7.text = "   - Decouples policy preferences (Actor) from state-value estimation (Critic).\n   - Softmax action selection allows for soft, probabilistic associations in high signal attenuation."
    set_font(p7.runs[0], size=16, color=c_dark)

    notes10 = slide10.notes_slide.notes_text_frame
    notes10.text = (
        "We benchmark three distinct algorithm classes representing tabular, continuous, and stochastic policy paradigms.\n\n"
        "Double Q-Learning serves as our robust tabular baseline, specifically selected to combat maximization bias. "
        "Linear Function Approximation uses multi-tiled coding to resolve continuous state coordinates, preventing boundary "
        "discretization issues. Actor-Critic models policy preferences directly via softmax, giving us a probabilistic tracker "
        "capable of maintaining tentative tracks during signal fading."
    )

    # Slide 11: Experimental Results - Croatia 2407_1 Metrics
    slide11 = add_slide("Experimental Results: Quantitative Metrics")
    
    # Left text / table column
    tb = slide11.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(6.5), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "Performance Metrics (Croatia 2407_1 Dataset):"
    set_font(p1.runs[0], size=18, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "• Linear FA achieves the highest cumulative reward (81,685) and good associations (8,815) due to continuous koordinat resolution.\n" \
             "• Double Q-Learning is extremely robust, achieving high track consistency with only 1.0% bad associations.\n" \
             "• Actor-Critic successfully resolves 2 dominant vessels but suffers from slightly higher duplicate spawns (468) under high noise."
    set_font(p2.runs[0], size=14, color=c_dark)
    p2.space_before = Pt(15)
    
    # Add a simple text-based table representing the metrics
    p3 = tf.add_paragraph()
    p3.text = "Agent | Reward | Good Assoc | Bad Assoc % | Vessels\n" \
             "---------------------------------------------------\n" \
             "Double Q | 78,595 | 8,648 | 1.0% | 3\n" \
             "Linear FA | 81,685 | 8,815 | 1.3% | 3\n" \
             "Actor-Critic | 74,531 | 8,449 | 1.7% | 2"
    set_font(p3.runs[0], size=12, bold=True, font_name="Consolas", color=c_navy)
    p3.space_before = Pt(15)
    
    # Right Image column
    if os.path.exists("report/images/rl_comparison_croatia_2407_1.png"):
        slide11.shapes.add_picture("report/images/rl_comparison_croatia_2407_1.png", Inches(7.2), Inches(1.8), width=Inches(5.6))

    notes11 = slide11.notes_slide.notes_text_frame
    notes11.text = (
        "Here are the quantitative results. The table shows that Linear Function Approximation performs the best, "
        "accumulating the highest cumulative reward and most good associations. This is because continuous tile coding "
        "allows the agent to track fine-grained frequency slides without getting stuck on bin boundaries.\n\n"
        "Double Q-Learning, however, is a close second and remains highly robust, with only 1.0% bad associations. "
        "Actor-Critic also performs well, but has slightly more duplicate spawns due to its stochastic policy exploring "
        "actions in borderline states."
    )

    # Slide 12: Training Convergence Profile
    slide12 = add_slide("Training Convergence Profile")
    
    # Left Image column
    if os.path.exists("report/images/convergence_combined_500.png"):
        slide12.shapes.add_picture("report/images/convergence_combined_500.png", Inches(0.5), Inches(1.8), width=Inches(6.0))
        
    # Right Image column
    if os.path.exists("report/images/convergence_individual_500.png"):
        slide12.shapes.add_picture("report/images/convergence_individual_500.png", Inches(6.8), Inches(1.8), width=Inches(6.0))

    notes12 = slide12.notes_slide.notes_text_frame
    notes12.text = (
        "Let's look at the convergence graphs over 500 episodes.\n\n"
        "On the left, we see the comparative policy convergence. Double Q-Learning converges extremely fast (within 15-20 episodes) "
        "and stabilizes. Linear FA also converges but exhibits minor, prolonged oscillations. On the right, we see the individual "
        "learning curves.\n\n"
        "Notice the high variance in training returns. This is caused by the 5% minimum exploration rate of the epsilon-greedy schedule. "
        "In a sequential tracking problem, one random exploratory action (like rejecting a stable track) will cause the track to collapse, "
        "forfeiting thousands of future reward points. However, when evaluated with epsilon=0 (the solid line), we see that the greedy "
        "policies are actually highly stable and optimal."
    )

    # Slide 13: Timeline and Normalization
    slide13 = add_slide("Absolute timelines \& Policy Normalization")
    
    # Left text column
    tb = slide13.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(6.0), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "Absolute Time timelines (HH:MM:SS):"
    set_font(p1.runs[0], size=18, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "• Mapped directly to timestamps parsed from raw WAV filenames.\n" \
             "• Double Q-Learning resolves the main ferry transit and support boat tracks without identity swaps."
    set_font(p2.runs[0], size=14, color=c_dark)
    
    p3 = tf.add_paragraph()
    p3.text = "Actor-Critic Policy Saturation Resolve:"
    set_font(p3.runs[0], size=18, bold=True, color=c_navy)
    p3.space_before = Pt(15)
    
    p4 = tf.add_paragraph()
    p4.text = "Early AC flatlined due to massive TD-errors saturating the softmax function. We resolved this using:\n" \
             "• Reward Scaling: Scaling rewards down by 20.0.\n" \
             "• Gradient Clipping: Clipping TD-error within [-1.0, 1.0].\n" \
             "• Preference Centering: Subtracting mean preferences at each step to prevent weights from drifting to infinity."
    set_font(p4.runs[0], size=14, color=c_dark)
    
    # Right Image column
    if os.path.exists("report/images/croatia_2407_1_double_q_learning_timeline.png"):
        slide13.shapes.add_picture("report/images/croatia_2407_1_double_q_learning_timeline.png", Inches(6.8), Inches(1.8), width=Inches(6.0))

    notes13 = slide13.notes_slide.notes_text_frame
    notes13.text = (
        "Here we discuss two critical topics: absolute timelines and Actor-Critic normalization.\n\n"
        "First, we integrated absolute time tracking. By parsing the timestamps in the raw file names, we map our sonar "
        "detections directly to real-world time in HH:MM:SS. The chart on the right shows that Double Q-Learning reconstructs the "
        "boats' timelines perfectly, matching passing ferry events recorded in field logs.\n\n"
        "Second, we resolved Actor-Critic flatlining. Early on, unscaled rewards led to massive TD-errors, causing the softmax "
        "preferences to saturate (one action probability goes to 1.0). We solved this by scaling the rewards down by 20.0, "
        "clipping TD-errors, and centering actor preferences at each step."
    )

    # Slide 14: Conclusion
    slide14 = add_slide("Conclusions \& Future Outlook")
    
    tb = slide14.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.33), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "Key Contributions of this Project:"
    set_font(p1.runs[0], size=22, bold=True, color=c_navy)
    
    p2 = tf.add_paragraph()
    p2.text = "• Formulation of passive acoustic tracking as an autonomous, self-supervised MDP.\n" \
             "• Successful elimination of human-labeled training data requirements by utilizing physical consistency constraints.\n" \
             "• Benchmarking of tabular, continuous, and stochastic policy models under a unified codebase.\n" \
             "• Integration of absolute real-world timestamps for time-synchronized trajectory reconstruction."
    set_font(p2.runs[0], size=16, color=c_dark)
    p2.space_before = Pt(15)
    
    p3 = tf.add_paragraph()
    p3.text = "Future Research Directions:"
    set_font(p3.runs[0], size=20, bold=True, color=c_navy)
    p3.space_before = Pt(20)
    
    p4 = tf.add_paragraph()
    p4.text = "• Reduce tile coding crosstalk (feature crosstalk interference) to stabilize continuous function approximation return profiles.\n" \
             "• Evaluate policy robustness under active acoustic jamming or extreme environmental surface clutter."
    set_font(p4.runs[0], size=16, color=c_dark)

    notes14 = slide14.notes_slide.notes_text_frame
    notes14.text = (
        "In conclusion, we have designed and validated a robust reinforcement learning framework for passive sonar vessel tracking.\n\n"
        "Our method eliminates the need for human labels, converting physical acoustic rules and temporal consistency into a "
        "self-supervised feedback loop.\n\n"
        "In the future, we plan to refine the continuous function approximation models to minimize feature crosstalk and evaluate "
        "robustness under active jamming or extreme marine storm noise."
    )

    # Save presentation
    output_path = "report/presentation_v2.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    create_presentation()
