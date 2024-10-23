from datetime import timedelta

def format_time(total_time):
    """
    Convert total_time in minutes to a human-readable format in Brazilian Portuguese.
    """
    # Convert total_time (in minutes) to a timedelta
    time_delta = timedelta(minutes=total_time)

    # Extract days, hours, and minutes
    days = time_delta.days
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes = remainder // 60

    # Build a human-readable string for Brazilian Portuguese
    if days > 0:
        human_readable_time = f"{days} dia{'s' if days > 1 else ''}, {hours} hora{'s' if hours > 1 else ''} e {minutes} minuto{'s' if minutes > 1 else ''}"
    else:
        human_readable_time = f"{hours} hora{'s' if hours > 1 else ''} e {minutes} minuto{'s' if minutes > 1 else ''}"

    return human_readable_time
